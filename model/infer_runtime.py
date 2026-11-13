"""推論実行エンジン — サーバー環境で利用可能な最高速のバックエンドで 3D-CNN を実行する。

本番サービスおよび評価スクリプトはモデルを直接呼び出さず、get_runner(model) が返却するランナー経由で推論を行う。
ランナーは logits を返却し、runner.with_cam(clips) は logits と位置別クラスマップ(CAM)を同時に
返却する。変換モデル(ONNX・OpenVINO)は双出力を単一推論で同時生成するため、CAM 有効時も高速推論を
そのまま維持可能（従来は logits のみ出力可能で CAM 使用時は PyTorch にフォールバックし CPU で約4倍低速化していた）。

バックエンド種別 (config.INFER_BACKEND):
  auto        : 利用可能なバックエンドを初回に自己診断し、検証をパスした最高速のものを自動選択(デフォルト)
  torch       : PyTorch ネイティブ実行。モデルが GPU 上にあれば GPU で実行。
  openvino    : OpenVINO (CPU) — Intel CPU 環境で大幅に高速化
  onnxruntime : ONNX Runtime (CPU)
モデルが GPU(CUDA)上にある場合、auto は PyTorch GPU を選択する(CPU バックエンドより高速)。FP16 は
config.INFER_FP16 が有効かつ Tensor Core 搭載の NVIDIA GPU において自己診断を通過した場合のみ適用される。

自己診断: 同一入力で PyTorch(FP32) と P(A)・logits・CAM を比較。未インストール・変換失敗・
許容誤差超過の場合は当該バックエンドを破棄し次の候補（最終的には PyTorch）へフォールバックする。そのため任意の OS・
ハードウェア環境で確実に機能し、出力結果は PyTorch と数学的に同等（FP32 精度維持）となる。

診断結果および変換ファイル(ONNX)は config.INFER_CACHE_DIR にキャッシュされる。重み・パッケージバージョン・
CPU/GPU・OS が同一であれば次回以降はベンチマークを再実行せず、サーバー起動時に1ウィンドウで
出力結果のみを PyTorch と高速照合（数秒）。初回セットアップ時の全診断は約数十秒を要する場合がある。
セットアップ直後に事前診断を実行する場合:  python -m model.main --mode check-backend
"""
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import config
from device_utils import is_cuda_like, predict_batch_size

# 診断・変換仕様の変更時にバージョンをインクリメント → キャッシュ・変換ファイルを自動再生成
# (2: 変換モデルが logits と共に位置別クラスマップ(CAM)も同時出力)
RUNTIME_VERSION = 2


class Runner:
    """推論ランナー。runner(clips (B, 3, T, H, W)) → logits (B, C) float32。
    runner.with_cam(clips) → (logits, cam (B, C, 7, 7)) — 単一の推論パスで同時算出。"""

    def __init__(self, name, fn, detail='', cam_fn=None):
        self.name = name        # 'torch' | 'torch-fp16' | 'openvino' | 'onnxruntime'
        self.detail = detail
        self._fn = fn
        self._cam_fn = cam_fn

    def __call__(self, clips):
        return self._fn(clips)

    def with_cam(self, clips):
        return self._cam_fn(clips)

    def __repr__(self):
        return f'Runner({self.name}{", " + self.detail if self.detail else ""})'


# ── 共通ユーティリティ ────────────────────────────────────────────────────────────

def _log(msg):
    print(f'[推論エンジン] {msg}', flush=True)


def _pkg_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def cpu_brand():
    """CPU ブランド名 — OS ごとに取得元が異なる（Linux /proc/cpuinfo, macOS sysctl, Windows レジストリ）。"""
    system = platform.system()
    try:
        if system == 'Linux':
            with open('/proc/cpuinfo', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if line.startswith('model name'):
                        return line.split(':', 1)[1].strip()
        elif system == 'Darwin':
            out = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                 capture_output=True, text=True, timeout=5).stdout.strip()
            if out:
                return out
        elif system == 'Windows':
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r'HARDWARE\DESCRIPTION\System\CentralProcessor\0')
            return str(winreg.QueryValueEx(key, 'ProcessorNameString')[0]).strip()
    except Exception:  # noqa: BLE001 — 名称取得に失敗しても動作に影響なし
        pass
    return platform.processor() or platform.machine()


def cpu_vendor():
    s = f'{cpu_brand()} {platform.processor()}'.lower()
    if 'intel' in s or 'genuineintel' in s:
        return 'intel'
    if 'amd' in s or 'authenticamd' in s:
        return 'amd'
    if 'apple' in s or (platform.system() == 'Darwin' and platform.machine() == 'arm64'):
        return 'apple'
    return 'other'


def _state_hash(model):
    """重みデータのフィンガープリント — 同一重みであれば変換ファイル・診断結果を再利用。"""
    h = hashlib.sha1()
    for k, v in model.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _check_input(clip_shape, n):
    g = torch.Generator().manual_seed(1234)
    return torch.randn((n, *clip_shape), generator=g)


def _compare(ref, out):
    """基準 logits vs 比較対象 logits → (P(A) 最大差分, logits 最大差分)"""
    ref, out = ref.float().cpu(), out.float().cpu()
    d_prob = float((F.softmax(ref, 1)[:, 1] - F.softmax(out, 1)[:, 1]).abs().max())
    return d_prob, float((ref - out).abs().max())


def _check_against(ref_runner, runner, x):
    """PyTorch 基準と推論結果の比較 → (P(A) 最大差分, logits・CAM 中の最大差分)。合否判定は _passes。"""
    ref, ref_cam = ref_runner.with_cam(x)
    out, cam = runner.with_cam(x)
    d_prob, d_logit = _compare(ref, out)
    d_cam = float((ref_cam.float().cpu() - cam.float().cpu()).abs().max())
    return d_prob, max(d_logit, d_cam)


def _passes(d_prob, d_logit):
    return d_prob <= config.INFER_CHECK_MAX_PROB_DIFF and d_logit <= 1e-2


def _time_per_clip(runner, x, reps=2):
    runner(x)                                   # 初回呼び出し（ウォームアップ）は計測除外
    sync = torch.cuda.synchronize if x.is_cuda else (lambda: None)
    sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        runner(x)
    sync()
    return (time.perf_counter() - t0) / (reps * x.shape[0])


# ── バックエンド別ランナー ───────────────────────────────────────────────────

def _torch_runner(model, fp16=False):
    dev = next(model.parameters()).device

    def run(x):
        with torch.inference_mode(), torch.autocast(dev.type, dtype=torch.float16, enabled=fp16):
            return model(x).float()

    def run_cam(x):
        with torch.inference_mode(), torch.autocast(dev.type, dtype=torch.float16, enabled=fp16):
            logits, cam = model.forward_with_cam(x)
            return logits.float(), cam.float()
    return Runner('torch-fp16' if fp16 else 'torch', run, str(dev), run_cam)


class _TimeMean(torch.nn.Module):
    """AdaptiveAvgPool3d((1, None, None)) と等価な演算（時間軸平均）。ONNX が動的 AvgPool3d を置換できないため使用。"""

    def forward(self, x):
        return x.mean(dim=2, keepdim=True)


class _WithCam(torch.nn.Module):
    """変換用ラッパー — forward_with_cam を forward として公開（出力2件: logits, cam）。"""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m.forward_with_cam(x)


def _export_copy(model):
    """変換用 CPU コピー — モデルオブジェクトを複製せず重みのみを転送して新規インスタンス化（出力: logits, cam）。"""
    from hitandrun_model import HitAndRun3DCNN
    pool = getattr(model, 'temporal_pool', 'avg')
    m = HitAndRun3DCNN(num_classes=model.head_conv.out_channels, temporal_pool=pool)
    m.load_state_dict({k: v.detach().cpu() for k, v in model.state_dict().items()})
    m.eval()
    if pool == 'avg':
        m.head_pool.pool = _TimeMean()      # top3・avgtop3 の TemporalPool は mean・topk のためそのまま変換可能
    return _WithCam(m).eval()


def _pool_tag():
    """ヘッド時間プーリングが avg 以外の場合に変換ファイル・キャッシュ名へ付与するサフィックス。"""
    pool = getattr(config, 'HEAD_TEMPORAL_POOL', 'avg')
    return '' if pool == 'avg' else f'_{pool}'


def _onnx_path(state_hash):
    return Path(config.INFER_CACHE_DIR) / f'x3d_{state_hash[:16]}{_pool_tag()}_v{RUNTIME_VERSION}.onnx'


def _ensure_onnx(model, state_hash, clip_shape):
    """ONNX 変換モデル — キャッシュが存在すればロード、なければ新規エクスポート（同一重みでは初回のみ実行）。"""
    path = _onnx_path(state_hash)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'{path.stem}.tmp{os.getpid()}.onnx')
    sample = torch.zeros((1, *clip_shape))
    kw = dict(input_names=['x'], output_names=['logits', 'cam'], opset_version=17,
              dynamic_axes={'x': {0: 'b'}, 'logits': {0: 'b'}, 'cam': {0: 'b'}})
    with warnings.catch_warnings(), torch.inference_mode():
        warnings.simplefilter('ignore')
        try:
            torch.onnx.export(_export_copy(model), (sample,), str(tmp), dynamo=False, **kw)
        except TypeError:          # dynamo 引数を持たない旧バージョン torch
            torch.onnx.export(_export_copy(model), (sample,), str(tmp), **kw)
    os.replace(tmp, path)          # 他プロセスが書き込み途中のファイルを誤読しないようアトミック置換
    return path


def _openvino_runner(model, state_hash, clip_shape):
    import openvino as ov
    core = ov.Core()
    try:
        om = core.read_model(str(_ensure_onnx(model, state_hash, clip_shape)))
    except Exception:  # noqa: BLE001 — ONNX 変換に失敗した場合は PyTorch モデルから直接変換を試行
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            om = ov.convert_model(_export_copy(model), example_input=torch.zeros((1, *clip_shape)))
    om.reshape({om.inputs[0]: ov.PartialShape([-1, *clip_shape])})
    # FP32 固定 — AMX 等を搭載した CPU では OpenVINO のデフォルトが BF16 となり数値差分が生じるのを防止
    cm = core.compile_model(om, 'CPU', {'PERFORMANCE_HINT': 'LATENCY',
                                        'INFERENCE_PRECISION_HINT': 'f32'})
    req = cm.create_infer_request()
    out, out_cam = cm.outputs[0], cm.outputs[1]     # 変換順序通り logits, cam

    def run_cam(x):
        r = req.infer({0: x.detach().cpu().float().contiguous().numpy()})
        # 出力バッファは次回呼び出しで上書きされるためコピーを保持
        return torch.from_numpy(np.array(r[out])), torch.from_numpy(np.array(r[out_cam]))

    def run(x):
        return run_cam(x)[0]    # CAM も同一推論で既に算出済み（1×1 乗算 49 グリッドのため追加コストは極小）
    return Runner('openvino', run, 'CPU f32', run_cam)


def _onnxruntime_runner(model, state_hash, clip_shape):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(_ensure_onnx(model, state_hash, clip_shape)), so,
                                providers=['CPUExecutionProvider'])

    def run(x):
        return torch.from_numpy(sess.run(['logits'], {'x': x.detach().cpu().float().contiguous().numpy()})[0])

    def run_cam(x):
        logits, cam = sess.run(['logits', 'cam'], {'x': x.detach().cpu().float().contiguous().numpy()})
        return torch.from_numpy(logits), torch.from_numpy(cam)
    return Runner('onnxruntime', run, 'CPU', run_cam)


_CPU_BUILDERS = {'openvino': _openvino_runner, 'onnxruntime': _onnxruntime_runner}


def _cpu_candidates():
    """CPU 候補の探索優先順位（同一速度であれば先頭を優先）。auto は全件ベンチマーク後に最高速を選択。"""
    if cpu_vendor() == 'intel':
        return ['openvino', 'onnxruntime']
    return ['onnxruntime', 'openvino']


# ── 自己診断結果キャッシュ ──────────────────────────────────────────────────

def _cache_file():
    return Path(config.INFER_CACHE_DIR) / 'backend_choice.json'


def _load_cache():
    try:
        return json.loads(_cache_file().read_text(encoding='utf-8'))
    except Exception:  # noqa: BLE001
        return {}


def _save_cache(key, entry):
    try:
        data = _load_cache()
        data[key] = entry
        f = _cache_file()
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_name(f'{f.name}.tmp{os.getpid()}')
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding='utf-8')
        os.replace(tmp, f)
    except Exception as exc:  # noqa: BLE001 — 保存失敗時も今回の推論実行自体は継続
        _log(f'自己診断結果の保存に失敗しました(次回起動時に再診断): {exc}')


def _fingerprint(state_hash, clip_shape, batch):
    return {
        'runtime_version': RUNTIME_VERSION, 'weights': state_hash[:16] + _pool_tag(),
        'clip_shape': list(clip_shape), 'batch': batch,
        'backend_cfg': (config.INFER_BACKEND or 'auto').lower(),
        'tol': config.INFER_CHECK_MAX_PROB_DIFF,
        'os': platform.system(), 'os_release': platform.release(), 'machine': platform.machine(),
        'cpu': cpu_brand(), 'torch': torch.__version__,
        'openvino': _pkg_version('openvino'), 'onnxruntime': _pkg_version('onnxruntime'),
    }


# ── 最適バックエンド選定 ──────────────────────────────────────────────────────────

def _fp16_supported(dev):
    """Tensor Core 搭載の NVIDIA GPU（演算能力 7.0 以上）のみ FP16 が高速化。AMD(ROCm) は除外。"""
    if dev.type != 'cuda' or torch.version.hip is not None:
        return False
    return torch.cuda.get_device_capability(dev) >= (7, 0)


def _select_gpu(model, clip_shape, batch):
    dev = next(model.parameters()).device
    # cuDNN は入力形状ごとに初回決定した計算カーネルをキャッシュし、後から benchmark を有効化しても再計測しない。
    # 下記の FP16 自己診断が本番バッチサイズで初回実行されるため、その前に有効化しておくことで最速カーネルが選定される。
    # (2026-10-22 fp16_check: 無効のまま先行実行され FP32 が 41.6 → 28.1窓/秒に縛られた事象に対応)。
    if is_cuda_like(dev):
        torch.backends.cudnn.benchmark = True
    base = _torch_runner(model)
    if not config.INFER_FP16:
        _log(f'PyTorch GPU ({torch.cuda.get_device_name(dev)}) — FP32')
        return base
    if not _fp16_supported(dev):
        _log(f'FP16 が有効化されましたが、このGPU({torch.cuda.get_device_name(dev)})はFP16高速化対象外のため FP32 で実行します')
        return base
    x = _check_input(clip_shape, batch).to(dev)
    half = _torch_runner(model, fp16=True)
    d_prob, _ = _compare(base(x), half(x))
    if d_prob <= config.INFER_FP16_MAX_PROB_DIFF:
        _log(f'PyTorch GPU ({torch.cuda.get_device_name(dev)}) — FP16 (自己診断通過: P(A) 差分 {d_prob:.1e})')
        return half
    _log(f'FP16 自己診断不合格(P(A) 差分 {d_prob:.1e} > {config.INFER_FP16_MAX_PROB_DIFF}) → FP32 で実行します')
    return base


def _select_cpu(model, clip_shape, batch, force=False):
    cfg = (config.INFER_BACKEND or 'auto').lower()
    if cfg == 'torch':
        _log('PyTorch CPU (config.INFER_BACKEND=torch)')
        return _torch_runner(model)
    if cfg not in ('auto', *_CPU_BUILDERS):
        _log(f"不明な INFER_BACKEND '{cfg}' → auto に切り替えます")
        cfg = 'auto'

    state_hash = _state_hash(model)
    fp = _fingerprint(state_hash, clip_shape, batch)
    key = hashlib.sha1(json.dumps(fp, sort_keys=True).encode()).hexdigest()

    # ① キャッシュされた診断結果が存在する場合、速度ベンチマークはスキップし、1ウィンドウで出力結果のみを高速検証。
    #    キャッシュ情報では検知できない環境変化（パッケージの破損等）もサーバー起動時に確実に排除。
    saved = None if force else _load_cache().get(key)
    if saved:
        name = saved['choice']
        try:
            if name == 'torch':
                runner, note = _torch_runner(model), ''
            else:
                runner = _CPU_BUILDERS[name](model, state_hash, clip_shape)
                x1 = _check_input(clip_shape, 1)
                d_prob, d_logit = _check_against(_torch_runner(model), runner, x1)
                if not _passes(d_prob, d_logit):
                    raise RuntimeError(f'クイック検証不合格(P(A) 差分 {d_prob:.1e}, logits・CAM 差分 {d_logit:.1e})')
                note = f', クイック検証合格(P(A) 差分 {d_prob:.1e})'
            _log(f"{name} — キャッシュされた診断結果を適用({saved.get('checked_at', '?')}, "
                 f"{saved.get('summary', '')}){note}")
            return runner
        except Exception as exc:  # noqa: BLE001
            _log(f'キャッシュされた方式 {name} が現環境と不整合のため再診断を実行します: {exc}')

    # ② 自己診断: PyTorch 出力を基準として各候補の数値精度比較 + 実行速度計測
    _log(f'この環境で初めて実行される構成のため自己診断を実行します(初回のみ) — CPU: {cpu_brand()}')
    x = _check_input(clip_shape, batch)
    ref_runner = _torch_runner(model)
    results = {'torch': {'ok': True, 'sec_per_clip': _time_per_clip(ref_runner, x)}}
    names = _cpu_candidates() if cfg == 'auto' else [cfg]
    built = {'torch': ref_runner}
    for name in names:
        try:
            runner = _CPU_BUILDERS[name](model, state_hash, clip_shape)
            d_prob, d_logit = _check_against(ref_runner, runner, x)
            ok = _passes(d_prob, d_logit)
            res = {'ok': ok, 'prob_diff': d_prob, 'logit_diff': d_logit}
            if ok:
                res['sec_per_clip'] = _time_per_clip(runner, x)
                built[name] = runner
            else:
                del runner
            results[name] = res
        except ModuleNotFoundError as exc:
            results[name] = {'ok': False, 'error': f'未インストール ({exc.name})'}
        except Exception as exc:  # noqa: BLE001 — 個別の失敗時は当該方式のみ破棄
            results[name] = {'ok': False, 'error': f'{type(exc).__name__}: {str(exc)[:200]}'}
        gc.collect()

    passed = {n: r for n, r in results.items() if r['ok']}
    if cfg == 'auto':
        choice = min(passed, key=lambda n: passed[n]['sec_per_clip'])
    else:
        choice = cfg if cfg in passed else 'torch'
    for n in list(built):
        if n != choice:
            del built[n]
    gc.collect()

    for n, r in results.items():
        if r['ok']:
            extra = (f", P(A) 差分 {r['prob_diff']:.1e}" if 'prob_diff' in r else ' (基準)')
            _log(f"  {n:12s} 合格  1窓 {r['sec_per_clip']:.3f}秒{extra}")
        else:
            why = r.get('error') or f"出力差分超過(P(A) 差分 {r['prob_diff']:.1e}, logits・CAM 差分 {r['logit_diff']:.1e})"
            _log(f'  {n:12s} 除外  {why}')
    t = results[choice]['sec_per_clip']
    summary = f"1窓 {t:.3f}秒, PyTorch比 ×{results['torch']['sec_per_clip'] / t:.2f}"
    _log(f'→ {choice} を選択 ({summary})')
    _save_cache(key, {'choice': choice, 'summary': summary,
                      'checked_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
                      'fingerprint': fp, 'results': results})
    return built[choice]


def get_runner(model, clip_shape=None, force_check=False):
    """model に適した推論ランナーを取得。初回のみ選定を行い model インスタンスにアタッチして再利用。

    force_check=True の場合はキャッシュを無視して再診断を実行(check-backend コマンド)。
    """
    clip_shape = tuple(clip_shape or (3, config.CLIP_LENGTH, config.RESIZE[1], config.RESIZE[0]))
    key = ((config.INFER_BACKEND or 'auto').lower(), bool(config.INFER_FP16), clip_shape)
    cached = model.__dict__.get('_infer_runner')
    if cached is not None and cached[0] == key and not force_check:
        return cached[1]
    model.eval()
    dev = next(model.parameters()).device
    batch = predict_batch_size(dev)     # サービスが実際に使用するバッチサイズで選定(CPU 1 · GPU 8)
    if dev.type == 'cpu':
        runner = _select_cpu(model, clip_shape, batch, force=force_check)
    else:
        runner = _select_gpu(model, clip_shape, batch)
    model.__dict__['_infer_runner'] = (key, runner)
    return runner
