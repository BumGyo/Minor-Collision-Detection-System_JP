"""推論ランナー — サーバーマシンで利用可能な最も高速なバックエンドで 3D-CNN を実行する。

本番サービスおよび評価ではモデルを直接呼び出さず、get_runner(model) が返すランナーを介して推論を行う。
ランナーは logits を返し、runner.with_cam(clips) は logits とグリッド別クラス活性化マップ (CAM) を同時に
返す。変換モデル (ONNX・OpenVINO) は1回の推論パスで両方の出力を同時に生成するため、CAM を有効にしても
高速なバックエンドをそのまま利用できる (以前は logits のみを出力していたため、CAM 有効時に全探索が PyTorch で
フォールバックされ、CPU 環境で約4倍低速化していた)。

バックエンド (config.INFER_BACKEND):
  auto        : 利用可能なバックエンドを初回に自己診断し、合格した中で最速のものを選択 (デフォルト)
  torch       : PyTorch そのまま。モデルが GPU 上にあれば GPU で実行。
  openvino    : OpenVINO (CPU) — Intel CPU 環境で高速
  onnxruntime : ONNX Runtime (CPU)
モデルが GPU(CUDA) にある場合、auto は PyTorch GPU を使用する (各種 CPU バックエンドより高速)。
FP16 は config.INFER_FP16 が有効かつ Tensor Core を搭載した NVIDIA GPU で自己診断に合格した場合のみ使用。

自己診断: 同一入力で PyTorch(FP32) と P(A)・logits・CAM を比較検証する。未インストール・変換失敗・
検証不合格の場合はそのバックエンドを除外し、次の候補 (最終的には PyTorch) へフォールバックする。
そのため、あらゆる OS・ハードウェア環境で確実に動作し、推論結果は PyTorch と一致する (FP32 精度を維持)。

診断結果および変換ファイル (ONNX) は config.INFER_CACHE_DIR にキャッシュされる。重み・パッケージバージョン・
CPU/GPU・OS が同一であれば、次回以降は速度ベンチマークをスキップし、サーバー (ワーカー) 起動時に1ウィンドウのみで
結果を PyTorch と高速照合する (数秒)。初回のみ全体診断に数十秒を要する場合がある。
インストール直後に事前診断を実行する場合:  python -m model.main --mode check-backend
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

# 診断・変換方式が変更された場合にインクリメント → 保存された結果・変換ファイルを再生成する
# (2: 変換モデルが logits と共にグリッド別クラス活性化マップ (CAM) も出力するように拡張)
RUNTIME_VERSION = 2


class Runner:
    """推論ランナー。runner(clips (B, 3, T, H, W)) → logits (B, C) float32。
    runner.with_cam(clips) → (logits, cam (B, C, 7, 7)) — 単一の推論パスで同時に返却。"""

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


# ── 衝突判定 (学習検証・評価・本番サービス共通) ───────────────────────────

def a_logit_margin(threshold=None):
    """P(A) 閾値 (config.A_THRESHOLD) をロジット差分 (logit_A − logit_S) のマージンに変換した値。0.5 → 0。"""
    t = float(config.A_THRESHOLD if threshold is None else threshold)
    if not 0.0 < t < 1.0:
        raise ValueError(f'A_THRESHOLD は 0 と 1 の間である必要があります: {t}')
    return float(np.log(t / (1.0 - t)))


def is_accident(logits, threshold=None):
    """(N, 2) logits → ウィンドウごとの衝突 (A) 判定 (N,) bool。P(A) > 閾値 と等価。

    ロジット差分で比較し、閾値 0.5 の場合は argmax (同値時は S) と完全に一致する。
    """
    return (logits[:, 1].float() - logits[:, 0].float()) > a_logit_margin(threshold)


# ── 共通ユーティリティ ──────────────────────────────────────────────────

def _log(msg):
    print(f'[推論ランナー] {msg}', flush=True)


def _pkg_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def cpu_brand():
    """CPU 名の取得 — OS ごとに取得元が異なる (Linux /proc/cpuinfo, macOS sysctl, Windows レジストリ)。"""
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
    except Exception:  # noqa: BLE001 — 名前が取得できなくても動作には支障なし
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
    """重みパラメータのフィンガープリント — 同一の重みであれば変換ファイル・診断結果を再利用する。"""
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
    """PyTorch 基準と結果を比較 → (P(A) 最大差分, logits・CAM のうち大きい方の最大差分)。合否は _passes。"""
    ref, ref_cam = ref_runner.with_cam(x)
    out, cam = runner.with_cam(x)
    d_prob, d_logit = _compare(ref, out)
    d_cam = float((ref_cam.float().cpu() - cam.float().cpu()).abs().max())
    return d_prob, max(d_logit, d_cam)


def _passes(d_prob, d_logit):
    return d_prob <= config.INFER_CHECK_MAX_PROB_DIFF and d_logit <= 1e-2


def _time_per_clip(runner, x, reps=2):
    runner(x)                                   # 初回呼び出し (ウォームアップ) を除外
    sync = torch.cuda.synchronize if x.is_cuda else (lambda: None)
    sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        runner(x)
    sync()
    return (time.perf_counter() - t0) / (reps * x.shape[0])


# ── バックエンド別ランナー ─────────────────────────────────────────────

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
    """AdaptiveAvgPool3d((1, None, None)) と等価な演算 (時間軸平均)。ONNX 変換の互換性対策。"""

    def forward(self, x):
        return x.mean(dim=2, keepdim=True)


class _WithCam(torch.nn.Module):
    """変換用ラッパー — forward_with_cam を forward として出力 (2つの出力: logits, cam)。"""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m.forward_with_cam(x)


def _export_copy(model):
    """変換用 CPU コピー — モデルオブジェクトを複製せず重みのみを移して新規作成 (出力: logits, cam)。"""
    from hitandrun_model import HitAndRun3DCNN
    pool = getattr(model, 'temporal_pool', 'avg')
    m = HitAndRun3DCNN(num_classes=model.head_conv.out_channels, temporal_pool=pool)
    m.load_state_dict({k: v.detach().cpu() for k, v in model.state_dict().items()})
    m.eval()
    if pool == 'avg':
        m.head_pool.pool = _TimeMean()      # top3・avgtop3 の TemporalPool は mean・topk なのでそのまま変換可能
    return _WithCam(m).eval()


def _pool_tag():
    """ヘッドの時間プーリングが avg でない場合、変換ファイル・診断ハッシュに付与する識別子。"""
    pool = getattr(config, 'HEAD_TEMPORAL_POOL', 'avg')
    return '' if pool == 'avg' else f'_{pool}'


def _onnx_path(state_hash):
    return Path(config.INFER_CACHE_DIR) / f'x3d_{state_hash[:16]}{_pool_tag()}_v{RUNTIME_VERSION}.onnx'


def _ensure_onnx(model, state_hash, clip_shape):
    """ONNX 変換ファイル — 存在すればそのまま返し、なければ生成する (同一の重みであれば1回のみ)。"""
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
        except TypeError:          # dynamo 引数を持たない古い torch
            torch.onnx.export(_export_copy(model), (sample,), str(tmp), **kw)
    os.replace(tmp, path)          # 他プロセスが書きかけのファイルを読み込まないようアトミックに置換
    return path


def _openvino_runner(model, state_hash, clip_shape):
    import openvino as ov
    core = ov.Core()
    try:
        om = core.read_model(str(_ensure_onnx(model, state_hash, clip_shape)))
    except Exception:  # noqa: BLE001 — ONNX の作成に失敗した場合は PyTorch モデルから直接変換
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            om = ov.convert_model(_export_copy(model), example_input=torch.zeros((1, *clip_shape)))
    om.reshape({om.inputs[0]: ov.PartialShape([-1, *clip_shape])})
    # FP32 固定 — AMX などを搭載した CPU では OpenVINO のデフォルトが BF16 となり結果に微差が生じる場合がある
    cm = core.compile_model(om, 'CPU', {'PERFORMANCE_HINT': 'LATENCY',
                                        'INFERENCE_PRECISION_HINT': 'f32'})
    req = cm.create_infer_request()
    out, out_cam = cm.outputs[0], cm.outputs[1]     # 変換順序通り logits, cam

    def run_cam(x):
        r = req.infer({0: x.detach().cpu().float().contiguous().numpy()})
        # 結果バッファは次の呼び出しで上書きされるためコピーして返す
        return torch.from_numpy(np.array(r[out])), torch.from_numpy(np.array(r[out_cam]))

    def run(x):
        return run_cam(x)[0]    # CAM も同一推論パスで計算済み (1×1 積 49 グリッドのためオーバーヘッドはごくわずか)
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
    """CPU 候補順序 (同速の場合は前者を優先)。auto は全件をベンチマークして最速を選択。"""
    if cpu_vendor() == 'intel':
        return ['openvino', 'onnxruntime']
    return ['onnxruntime', 'openvino']


# ── 診断結果の保存 ──────────────────────────────────────────────────

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
    except Exception as exc:  # noqa: BLE001 — 保存失敗時も今回の実行自体は継続
        _log(f'自己診断結果を保存できませんでした (次回実行時に再診断): {exc}')


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


# ── バックエンド選択 ──────────────────────────────────────────────────

def _fp16_supported(dev):
    """Tensor Core を搭載した NVIDIA GPU (演算能力 7.0 以上) のみ FP16 による高速化が有効。AMD(ROCm) は除外。"""
    if dev.type != 'cuda' or torch.version.hip is not None:
        return False
    return torch.cuda.get_device_capability(dev) >= (7, 0)


def _select_gpu(model, clip_shape, batch):
    dev = next(model.parameters()).device
    # cuDNN は入力形状ごとに最初に決定した演算アルゴリズムを記憶し、後から benchmark を有効にしても再測定しない。
    # 下記の FP16 自己診断が本番バッチサイズでモデルを初回実行するため、その前に有効化しておく必要がある。
    if is_cuda_like(dev):
        torch.backends.cudnn.benchmark = True
    base = _torch_runner(model)
    if not config.INFER_FP16:
        _log(f'PyTorch GPU ({torch.cuda.get_device_name(dev)}) — FP32')
        return base
    if not _fp16_supported(dev):
        _log(f'FP16 が有効ですが、この GPU ({torch.cuda.get_device_name(dev)}) は FP16 による高速化が期待できないため FP32 で実行します')
        return base
    x = _check_input(clip_shape, batch).to(dev)
    half = _torch_runner(model, fp16=True)
    d_prob, _ = _compare(base(x), half(x))
    if d_prob <= config.INFER_FP16_MAX_PROB_DIFF:
        _log(f'PyTorch GPU ({torch.cuda.get_device_name(dev)}) — FP16 (自己診断合格: P(A) 差分 {d_prob:.1e})')
        return half
    _log(f'FP16 自己診断不合格 (P(A) 差分 {d_prob:.1e} > {config.INFER_FP16_MAX_PROB_DIFF}) → FP32 で実行します')
    return base


def _select_cpu(model, clip_shape, batch, force=False):
    cfg = (config.INFER_BACKEND or 'auto').lower()
    if cfg == 'torch':
        _log('PyTorch CPU (config.INFER_BACKEND=torch)')
        return _torch_runner(model)
    if cfg not in ('auto', *_CPU_BUILDERS):
        _log(f"不明な INFER_BACKEND '{cfg}' → auto で選択します")
        cfg = 'auto'

    state_hash = _state_hash(model)
    fp = _fingerprint(state_hash, clip_shape, batch)
    key = hashlib.sha1(json.dumps(fp, sort_keys=True).encode()).hexdigest()

    # ① キャッシュされた診断結果が存在する場合はベンチマーク測定をスキップし、1ウィンドウのみで結果を高速確認する。
    #    環境情報の変化 (パッケージの破損等) もサーバー起動時に検知可能。
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
                    raise RuntimeError(f'簡易チェック不合格 (P(A) 差分 {d_prob:.1e}, logits・CAM 差分 {d_logit:.1e})')
                note = f', 簡易チェック合格 (P(A) 差分 {d_prob:.1e})'
            _log(f"{name} — 保存された自己診断結果を使用 ({saved.get('checked_at', '?')}, "
                 f"{saved.get('summary', '')}){note}")
            return runner
        except Exception as exc:  # noqa: BLE001
            _log(f'保存された方式 {name} が現在の環境と一致しないため再診断を実行します: {exc}')

    # ② 自己診断: PyTorch の出力を基準として各候補の出力比較および速度測定を実施
    _log(f'このマシンで初回実行の構成のため自己診断を実行します (初回のみ) — CPU: {cpu_brand()}')
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
        except Exception as exc:  # noqa: BLE001 — 例外発生時はそのバックエンドのみ除外
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
            _log(f"  {n:12s} 合格  ウィンドウ1個 {r['sec_per_clip']:.3f}秒{extra}")
        else:
            why = r.get('error') or f"結果相違 (P(A) 差分 {r['prob_diff']:.1e}, logits・CAM 差分 {r['logit_diff']:.1e})"
            _log(f'  {n:12s} 除外  {why}')
    t = results[choice]['sec_per_clip']
    summary = f"ウィンドウ1個 {t:.3f}秒, PyTorch 比 ×{results['torch']['sec_per_clip'] / t:.2f}"
    _log(f'→ {choice} を選択 ({summary})')
    _save_cache(key, {'choice': choice, 'summary': summary,
                      'checked_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
                      'fingerprint': fp, 'results': results})
    return built[choice]


def get_runner(model, clip_shape=None, force_check=False):
    """model に対応する推論ランナーを返す。初回のみ選択を行い、model にキャッシュして再利用する。

    force_check=True の場合、キャッシュを無視して再診断を実施 (check-backend コマンド用)。
    """
    clip_shape = tuple(clip_shape or (3, config.CLIP_LENGTH, config.RESIZE[1], config.RESIZE[0]))
    key = ((config.INFER_BACKEND or 'auto').lower(), bool(config.INFER_FP16), clip_shape)
    cached = model.__dict__.get('_infer_runner')
    if cached is not None and cached[0] == key and not force_check:
        return cached[1]
    model.eval()
    dev = next(model.parameters()).device
    batch = predict_batch_size(dev)     # サービスが実際に使用するバッチサイズで選択 (CPU 1 ・ GPU 8)
    if dev.type == 'cpu':
        runner = _select_cpu(model, clip_shape, batch, force=force_check)
    else:
        runner = _select_gpu(model, clip_shape, batch)
    model.__dict__['_infer_runner'] = (key, runner)
    return runner
