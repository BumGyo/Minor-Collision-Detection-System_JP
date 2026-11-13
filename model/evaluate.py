import os
from datetime import datetime
import sys
import csv
import pathlib
import platform
import statistics
import time
import cv2
import torch
import numpy as np
from tqdm import tqdm

import config
from device_utils import is_cuda_like, is_channels_last_3d_supported, predict_batch_size
from infer_runtime import get_runner
from runlog import make_run_dir, write_run_info


def _evaluate_folder_impl(
    model,
    folder_path=config.EVAL_FOLDER_PATH,
    target_id=config.TARGET_ID,
    r_value=config.R_VALUE,
    resize=config.RESIZE,
    clip_length=config.CLIP_LENGTH,
    infer_batch_size=config.EVAL_INFER_BATCH_SIZE,
    window_stride=config.EVAL_WINDOW_STRIDE,
):
    folder_path = str(folder_path)
    print(f"[{folder_path}] フォルダ内の動画評価の準備中...")
    device = next(model.parameters()).device
    model.eval()
    window_stride = max(1, int(window_stride))
    # サービスと同一の推論ランナー (infer_runtime.py) を使用して評価結果・速度の完全一致を担保
    runner = get_runner(model)

    if is_cuda_like(device):
        torch.backends.cudnn.benchmark = True

    # 正規化統計量は学習時と同一に設定 (Kinetics-400 共通設定)
    mean = torch.tensor(config.NORM_MEAN,
                        dtype=torch.float32).view(3, 1, 1, 1)
    std = torch.tensor(config.NORM_STD,
                       dtype=torch.float32).view(3, 1, 1, 1)

    def crop_square_and_pad(frame, bbox, r):
        h, w, _ = frame.shape
        x_min, y_min, x_max, y_max = bbox
        vw, vh = (x_max - x_min) * r, (y_max - y_min) * r
        cx, cy = x_min + (x_max - x_min) // 2, y_min + (y_max - y_min) // 2
        side = int(max(vw, vh))
        nx1, ny1 = cx - side // 2, cy - side // 2
        nx2, ny2 = cx + side // 2, cy + side // 2
        v_x1, v_y1 = max(0, nx1), max(0, ny1)
        v_x2, v_y2 = min(w, nx2), min(h, ny2)
        p_l, p_t = max(0, -nx1), max(0, -ny1)
        p_r, p_b = max(0, nx2 - w), max(0, ny2 - h)
        cropped = frame[v_y1:v_y2, v_x1:v_x2]
        if p_l > 0 or p_t > 0 or p_r > 0 or p_b > 0:
            cropped = np.pad(
                cropped, ((p_t, p_b), (p_l, p_r), (0, 0)), mode='constant')
        return cv2.resize(cropped, resize, interpolation=cv2.INTER_LINEAR)

    def frames_to_tensor(frames):
        arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
        video_tensor = torch.from_numpy(arr).permute(3, 0, 1, 2).contiguous()
        return (video_tensor - mean) / std

    all_files = os.listdir(folder_path)
    all_files_set = set(all_files)
    mp4_files = [f for f in all_files if f.endswith('.mp4')]
    valid_pairs = [
        mp4 for mp4 in mp4_files if f"{mp4.rsplit('.', 1)[0]}.txt" in all_files_set]
    # 10fps 未満の動画 (config.EXCLUDE_VIDEOS) は評価から除外
    excl = set(getattr(config, 'EXCLUDE_VIDEOS', ()) or ())
    skipped = sorted(m.rsplit('.', 1)[0] for m in valid_pairs if m.rsplit('.', 1)[0] in excl)
    if skipped:
        valid_pairs = [m for m in valid_pairs if m.rsplit('.', 1)[0] not in excl]
        print(f"[除外] 10fps 未満 (config.LOW_FPS_VIDEOS) {len(skipped)}本は評価対象外: {', '.join(skipped)}")

    if not valid_pairs:
        print("評価可能な動画・テキストのペアが存在しません。")
        return

    # data/eval の有効な全ペアをソート順で評価
    selected_files = sorted(valid_pairs)
    print(f"data/eval の有効な全 {len(selected_files)}本の動画ペアを評価します。")

    total_videos = 0
    correct_preds = 0
    tp = tn = fp = fn = 0   # 混同行列 (Positive = A/衝突)
    wrong_list = []
    base_rows = {}          # 動画別 ① 判定

    with torch.inference_mode():
        for mp4_file in tqdm(selected_files, desc="評価進捗"):
            base_name = mp4_file.rsplit('.', 1)[0]
            video_path = os.path.join(folder_path, mp4_file)
            txt_path = os.path.join(folder_path, f"{base_name}.txt")

            parts = base_name.split('_')
            # クラス識別文字は parts[1] の「末尾文字」で判別する。
            #   rc: LA/RA/SA/NA(2文字、末尾=A) → 衝突、LS/LV/LW 等 → 非衝突
            #   real: ReA(3文字、末尾=A) → 衝突、ReS → 非衝突
            # (TODO: 将来的にファイル名の代わりにtxtの A/S action行から読み取るよう統一予定)
            is_accident_gt = len(parts) >= 2 and len(
                parts[1]) >= 2 and parts[1][-1] == 'A'
            gt_label = 1 if is_accident_gt else 0

            bboxes = {}
            with open(txt_path, 'r') as f:
                for line in f:
                    l_parts = line.strip().split(',')
                    if len(l_parts) >= 6 and l_parts[0] == 'car':
                        bboxes[int(l_parts[1])] = [int(l_parts[2]), int(
                            l_parts[3]), int(l_parts[4]), int(l_parts[5])]

            if target_id not in bboxes:
                if not bboxes:
                    continue
                target_bbox = next(iter(bboxes.values()))
            else:
                target_bbox = bboxes[target_id]

            cap = cv2.VideoCapture(video_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(crop_square_and_pad(
                    frame_rgb, target_bbox, r_value))
            cap.release()

            while len(frames) < clip_length:
                frames.append(
                    frames[-1] if frames else np.zeros((resize[1], resize[0], 3), dtype=np.uint8))

            # ⑦ 動画テンソルを最初からGPU上に常駐 — バッチごとの .to(device) 転送を削減
            full_video_tensor = frames_to_tensor(frames).to(device)
            predicted_label = 0
            first_a = None          # P(A) > 0.5 が初めて出現したウィンドウの開始フレーム
            num_windows = len(frames) - (clip_length - 1)
            window_starts = list(range(0, num_windows, window_stride))

            for batch_start in range(0, len(window_starts), infer_batch_size):
                batch_window_starts = window_starts[batch_start:batch_start +
                                                    infer_batch_size]
                # 既にGPU上に存在するテンソルをスライシング — .to() 呼び出しなし
                clips = torch.stack(
                    [full_video_tensor[:, i:i+clip_length, :, :] for i in batch_window_starts])
                if is_channels_last_3d_supported(device) and config.USE_CHANNELS_LAST:
                    clips = clips.contiguous(
                        memory_format=torch.channels_last_3d)

                outputs = runner(clips)
                pred_classes = outputs.argmax(dim=1)
                if (pred_classes == 1).any().item():
                    predicted_label = 1
                    first_a = batch_window_starts[int((pred_classes == 1).nonzero()[0])]
                    break

            total_videos += 1
            base_rows[mp4_file] = {'pred': 'A' if predicted_label else 'S', 'first_a': first_a}
            # 混同行列の集計 (Positive = A/衝突)
            if gt_label == 1 and predicted_label == 1:
                tp += 1
            elif gt_label == 0 and predicted_label == 0:
                tn += 1
            elif gt_label == 0 and predicted_label == 1:
                fp += 1
            else:  # gt_label == 1 and predicted_label == 0
                fn += 1

            if predicted_label == gt_label:
                correct_preds += 1
            else:
                wrong_list.append({
                    "file": mp4_file,
                    "gt": "Accident(衝突)" if gt_label == 1 else "Normal(正常)",
                    "pred": "Accident(衝突)" if predicted_label == 1 else "Normal(正常)",
                })

    # ── 論文 (Hwang & Lee 2024, Table 6) 準拠指標 ──────────────
    base = _metrics(tp, fn, fp, tn)
    print("\n" + "=" * 60)
    print("[① 従来判定]  ウィンドウ1つでも P(A) > 0.5 ならその動画は衝突  (論文 Table 6 方式)")
    print(f"   ウィンドウ間隔 {window_stride} で全ウィンドウを走査。粗探索・イベント統合ルールはなく、"
          "衝突位置は考慮しない。")
    print("=" * 60)
    _print_metrics(base)
    print("=" * 60)

    if wrong_list:
        print("\n[誤答分析 (誤判定動画リスト)]")
        for w in wrong_list:
            print(f" - {w['file']} (正解: {w['gt']}  |  モデル予測: {w['pred']})")
    else:
        print("\nすべての動画で完全一致しました！")

    # 本番サービスと同一の推論・後処理パイプラインで再度評価
    svc = evaluate_service_path(model, folder_path, selected_files,
                                target_id=target_id, r_value=r_value,
                                resize=resize, clip_length=clip_length,
                                base_rows=base_rows)
    svc_rows = svc.pop('rows')
    _print_comparison(base, svc, svc_rows, window_stride)

    # 推論速度: ① モデル単体 ② サービス経路全体
    speed = {'device': f'{_device_name(device)} ・ 実行方式 {runner.name}', 'model': benchmark_model(
        model, clip_length=clip_length, resize=resize), 'service': svc.pop('speed'),
        'service_batch': predict_batch_size(device)}
    _print_speed(speed)
    sb = {b['batch']: b for b in speed['model']}
    svc_b = sb.get(speed['service_batch'], speed['model'][-1])
    s = speed['service'] or {}
    return {
        'n_videos': total_videos,
        **{f'base_{k}': v for k, v in base.items()},
        **{(k if k.startswith(('clean_', 'n_in')) else f'svc_{k}'): v
           for k, v in svc.items()},
        'device': speed['device'],
        'speed_proc_fps': round(s['proc_fps'], 2) if s else None,
        'speed_realtime_x': round(s['realtime_x'], 3) if s else None,
        'speed_windows_eval_pct': round(s['windows_eval_pct'], 1) if s else None,
        'speed_model_batch': svc_b['batch'],
        'speed_model_clips_per_sec': svc_b['clips_per_sec'],
        'speed_model_ms_per_clip': svc_b['ms_per_clip'],
        '_speed': speed,
        '_svc_rows': svc_rows,
    }


def _metrics(tp, fn, fp, tn):
    """動画単位の混同行列 → 論文 Table 6 指標(%)。分母が0の場合はNone。

    Positive = A(衝突)。Recall = TP/(TP+FN), False alarm = FP/(FP+TN),
    Accuracy = (TP+TN)/全体, Precision = TP/(TP+FP), F1 = 2PR/(P+R)。
    """
    def pct(a, b):
        return round(a / b * 100, 2) if b else None

    rec, prec = pct(tp, tp + fn), pct(tp, tp + fp)
    if rec is None or prec is None:
        f1 = None
    else:
        f1 = round(2 * prec * rec / (prec + rec), 2) if (prec + rec) else 0.0
    return {'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
            'recall': rec, 'false_alarm': pct(fp, fp + tn),
            'accuracy': pct(tp + tn, tp + fn + fp + tn), 'precision': prec, 'f1': f1}


def _fmt_pct(v):
    return '-' if v is None else f'{v:.2f}%'


def _print_metrics(mt):
    print(f"動画数             : {mt['tp'] + mt['fn'] + mt['fp'] + mt['tn']} 本  "
          f"(A {mt['tp'] + mt['fn']} / S {mt['fp'] + mt['tn']})")
    print(f"混同行列           : TP {mt['tp']} / FN {mt['fn']} / FP {mt['fp']} / TN {mt['tn']}")
    print(f"Recall(再現率)      : {_fmt_pct(mt['recall'])}   (実際の衝突動画のうち衝突と判定)")
    print(f"False alarm(誤検知率): {_fmt_pct(mt['false_alarm'])}   (非衝突動画のうち衝突と誤判定)")
    print(f"Accuracy(正解率)    : {_fmt_pct(mt['accuracy'])}")
    print(f"Precision(適合率)   : {_fmt_pct(mt['precision'])}")
    print(f"F1-score            : {_fmt_pct(mt['f1'])}")


def _print_comparison(base, svc, rows, window_stride):
    """① 従来判定と ② サービス判定を同一フォーマットで並列比較。"""
    tol = svc.get('hit_tol_sec', 1.0)
    W = 22

    def line(name, a, b):
        print(f"{name:<{W}}{a:>18}{b:>22}")

    print("\n" + "=" * 64)
    print("[判定方式の比較]  同一動画、動画単位 (Positive = A/衝突)")
    print("=" * 64)
    line('', '① 従来判定', '② サービス判定')
    line('判定ルール', 'ウィンドウ P(A)>0.5 1個↑', 'イベント 1個↑')
    line('走査方式', f'間隔 {window_stride}, 全件', f'粗探索 → 候補区間のみ')
    print("-" * 64)
    line('TP / FN / FP / TN', f"{base['tp']}/{base['fn']}/{base['fp']}/{base['tn']}",
         f"{svc['tp']}/{svc['fn']}/{svc['fp']}/{svc['tn']}")
    line('Recall(再現率)', _fmt_pct(base['recall']), _fmt_pct(svc['video_recall']))
    line('False alarm(誤検知率)', _fmt_pct(base['false_alarm']), _fmt_pct(svc['video_false_alarm']))
    line('Accuracy(正解率)', _fmt_pct(base['accuracy']), _fmt_pct(svc['accuracy']))
    line('Precision(適合率)', _fmt_pct(base['precision']), _fmt_pct(svc['precision']))
    line('F1-score', _fmt_pct(base['f1']), _fmt_pct(svc['f1']))
    print("-" * 64)
    line(f'Recall(開始 ±{tol:g}秒)', '(位置考慮なし)', _fmt_pct(svc['hit_recall']))
    fph = svc.get('false_per_hour')
    line('誤検知イベント', '(イベントなし)',
         f"{svc['false_events']}個" + (f" (1時間あたり {fph})" if fph is not None else ''))
    print("=" * 64)
    print("  ・ 上段の指標は両判定とも「衝突位置に関わらず動画内に衝突が検出されたか」のみを評価。")
    print(f"  ・ サービス実用上の有効検出は「Recall(開始 ±{tol:g}秒)」です — 衝突開始近傍で")
    print("    開始したイベントのみをカウントします (無関係な位置のイベントは無駄足となる誤検知)。")
    diff = [r for r in rows if r.get('base_pred') and r['base_pred'] != r['svc_pred']]
    if diff:
        print(f"\n[両判定で結果が異なる動画 {len(diff)}本]")
        for r in diff:
            fa = f" (初窓A {r['base_first_a']}f)" if r.get('base_first_a') is not None else ''
            print(f" - {r['file']} (正解 {r['gt']}) : ① {r['base_pred']}{fa} / "
                  f"② {r['svc_pred']} (イベント {r['n_events']}個)")
    else:
        print("\n両判定で結果が異なる動画はありません。")


def _device_name(device):
    """実行環境のハードウェア識別文字列を生成。"""
    if is_cuda_like(device):
        try:
            return f"{device} ({torch.cuda.get_device_name(device)})"
        except Exception:  # noqa: BLE001
            return str(device)
    cpu = platform.processor() or platform.machine()
    try:
        with open('/proc/cpuinfo', encoding='utf-8') as f:
            cpu = next(l.split(':', 1)[1].strip() for l in f if l.startswith('model name'))
    except (OSError, StopIteration):
        pass
    return f"cpu ({cpu}, torch スレッド数 {torch.get_num_threads()})"


def benchmark_model(model, clip_length=config.CLIP_LENGTH, resize=config.RESIZE,
                    batch_sizes=None, min_sec=3.0, max_iters=30):
    """3D-CNN純粋推論速度 — 動画デコード・クロップ・前処理を除外してモデル単体を測定(サービスと同一の実行方式)。

    サービスはウィンドウ間隔1(PREDICT_WINDOW_STRIDE)のため、動画全体を漏れなく走査するにはフレーム
    1枚ごとにウィンドウ(30フレームクリップ)1個を推論する必要がある。そのため「秒あたり処理ウィンドウ数」がそのまま
    「モデルが秒あたり分析可能な動画フレーム数」となる(粗い探索なしで全件評価時)。
    """
    device = next(model.parameters()).device
    batch_sizes = batch_sizes or sorted({1, predict_batch_size(device),
                                         int(config.EVAL_INFER_BATCH_SIZE)})
    sync = (torch.cuda.synchronize if is_cuda_like(device) else (lambda: None))
    out = []
    runner = get_runner(model)
    with torch.inference_mode():
        for bs in batch_sizes:
            x = torch.randn(bs, 3, clip_length, resize[1], resize[0], device=device)
            if is_channels_last_3d_supported(device) and config.USE_CHANNELS_LAST:
                x = x.contiguous(memory_format=torch.channels_last_3d)
            for _ in range(2):                       # 初回ウォームアップ (cuDNN アルゴリズム探索等)
                runner(x)
            sync()
            n, t0 = 0, time.perf_counter()
            while n < max_iters and (time.perf_counter() - t0) < min_sec:
                runner(x)
                n += 1
            sync()
            sec = time.perf_counter() - t0
            clips_per_sec = n * bs / sec
            out.append({'batch': bs, 'clips_per_sec': round(clips_per_sec, 2),
                        'ms_per_clip': round(1000 / clips_per_sec, 2)})
    return out


def _print_speed(speed):
    print("\n" + "=" * 60)
    print("[推論速度]  デバイス: " + speed['device'])
    print("=" * 60)
    print("① モデル単体 (3D-CNN, 30フレームクリップ1個 = 窓1個)")
    for b in speed['model']:
        mark = '  ← サービスバッチ' if b['batch'] == speed.get('service_batch') else ''
        print(f"   バッチ {b['batch']:>2}: 秒あたり {b['clips_per_sec']:7.2f}個の窓 "
              f"(窓1個 {b['ms_per_clip']:.1f}ms){mark}")
    print("   → ウィンドウ間隔 1 で漏れなく走査した場合にモデルが秒あたり処理可能な動画フレーム数に相当")
    s = speed.get('service')
    if s:
        print("② サービス経路全体 (デコード・クロップ + モーション計算 + 粗探索・詳細推論, クリップ生成除外)")
        print(f"   動画 {s['n_videos']}本, {s['video_sec']:.1f}秒分 ({s['frames']}フレーム) → "
              f"処理 {s['proc_sec']:.1f}秒")
        print(f"   処理速度  : 秒あたり {s['proc_fps']:.1f} フレーム  "
              f"(動画1秒を {s['sec_per_video_sec']:.2f}秒で処理, 実時間の {s['realtime_x']:.2f}倍速)")
        print(f"   3D-CNN 推論窓: 全体 {s['windows_total']}個中 {s['windows_eval']}個 "
              f"({s['windows_eval_pct']:.0f}%, 粗探索により削減), 秒あたり {s['windows_per_sec']:.1f}個")
        print(f"   時間内訳  : デコード・クロップ {s['pct_decode']:.0f}% / モーション計算 {s['pct_flow']:.0f}% / "
              f"3D-CNN {s['pct_infer']:.0f}% / その他 {s['pct_other']:.0f}%")
        print(f"   動画別処理速度: 中央値 秒あたり {s['proc_fps_median']:.1f} フレーム "
              f"(最低 {s['proc_fps_min']:.1f} ・ 最高 {s['proc_fps_max']:.1f})")
    print("=" * 60)


def _speed_summary(rows):
    """動画別処理時間 → 全体処理速度の集計。"""
    rs = [r for r in rows if r.get('proc_sec')]
    if not rs:
        return None
    frames = sum(r['frames_read'] for r in rs)
    vsec = sum(r['frames_read'] / r['fps'] for r in rs if r['fps'])
    proc = sum(r['proc_sec'] for r in rs)
    dec = sum(r['decode_sec'] for r in rs)
    flow = sum(r['flow_sec'] for r in rs)
    inf = sum(r['infer_sec'] for r in rs)
    wt = sum(r['windows_total'] for r in rs)
    we = sum(r['windows_eval'] for r in rs)
    per = sorted(r['proc_fps'] for r in rs)
    pct = lambda x: x / proc * 100 if proc else 0.0  # noqa: E731
    return {
        'n_videos': len(rs), 'frames': frames, 'video_sec': vsec, 'proc_sec': proc,
        'proc_fps': frames / proc if proc else 0.0,
        'realtime_x': vsec / proc if proc else 0.0,
        'sec_per_video_sec': proc / vsec if vsec else 0.0,
        'windows_total': wt, 'windows_eval': we,
        'windows_eval_pct': we / wt * 100 if wt else 0.0,
        'windows_per_sec': we / inf if inf else 0.0,
        'pct_decode': pct(dec), 'pct_flow': pct(flow), 'pct_infer': pct(inf),
        'pct_other': max(0.0, 100 - pct(dec) - pct(flow) - pct(inf)),
        'proc_fps_median': statistics.median(per), 'proc_fps_min': per[0], 'proc_fps_max': per[-1],
    }


def _read_label(txt_path):
    """アノテーション txt → (bboxes, gt_start_f, gt_end_f)。A行が存在しない場合は (.., None, None)。"""
    bboxes, sf, ef = {}, None, None
    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            p = [t.strip() for t in line.strip().split(',')]
            if len(p) >= 6 and p[0] == 'car':
                try:
                    bboxes[int(p[1])] = [int(float(v)) for v in p[2:6]]
                except ValueError:
                    pass
            elif p and p[0] == 'A' and len(p) > 2:
                try:
                    sf = int(float(p[2]))
                    ef = int(float(p[3])) if len(p) > 3 else sf
                except ValueError:
                    pass
    return bboxes, sf, ef


def _training_video_names():
    """学習データフォルダ(config.DATA_DIR)配下の動画ファイル名セット。

    評価動画がここに含まれる場合、モデルが学習済み(train分割)または学習中の検証に使用(val
    分割)されている可能性があり性能が見かけ上高くなる。実際に eval 24本中18本が学習に含まれた状態で
    評価された事例があるため、ファイル名による重複を検知して警告し、未学習動画のみを別途集計する。
    """
    # 検証フォルダ(config.VAL_DATA_DIR, 2026-11-04〜)も学習中に使用する動画のため併せて確認
    names = set()
    for r in (config.DATA_DIR, getattr(config, 'VAL_DATA_DIR', None)):
        root = pathlib.Path(str(r)) if r else None
        if root is not None and root.exists():
            names |= {p.name for p in root.rglob('*.mp4')}
    return names


def _summarize(rows):
    """動画別結果(rows) → サービス判定指標。

    · 動画単位(① 従来判定と同一定義): イベントが1つでも発生すればその動画は衝突。
    · 位置基準: ラベル開始 ± 許容誤差以内で開始したイベントが存在すれば検出(hit_recall)。
    """
    A = [r for r in rows if r['gt'] == 'A']
    S = [r for r in rows if r['gt'] == 'S']
    hits = sum(1 for r in A if r['hit'])
    errs = sorted(r['err_f'] for r in A if r['err_f'] is not None)
    errs_s = sorted(r['err_sec'] for r in A if r['err_sec'] is not None)
    hours = sum(r['frames'] / r['fps'] for r in rows if r['fps']) / 3600

    def pct(a, b):
        return round(a / b * 100, 2) if b else None

    tp = sum(1 for r in A if r['n_events'])
    fp = sum(1 for r in S if r['n_events'])
    conf = _metrics(tp, len(A) - tp, fp, len(S) - fp)
    false_events = sum(r['false_events'] for r in rows)
    return {
        'n': len(rows), 'n_A': len(A), 'n_S': len(S),
        'tp': conf['tp'], 'fn': conf['fn'], 'fp': conf['fp'], 'tn': conf['tn'],
        'video_recall': conf['recall'], 'video_false_alarm': conf['false_alarm'],
        'accuracy': conf['accuracy'], 'precision': conf['precision'], 'f1': conf['f1'],
        'hit_recall': pct(hits, len(A)),
        'overlap_recall': pct(sum(1 for r in A if r['overlap']), len(A)),
        'missed': sum(1 for r in A if r['n_events'] == 0),
        'wrong_place': sum(1 for r in A if r['n_events'] and not r['hit']),
        'split': sum(1 for r in A if r['hit'] and r['n_events'] > 1),
        'false_events': false_events,
        'false_per_hour': round(false_events / hours, 1) if hours else None,
        'err_median_f': errs[len(errs) // 2] if errs else None,
        'err_median_sec': round(errs_s[len(errs_s) // 2], 3) if errs_s else None,
        'hours': hours,
    }


def _print_summary(s, tol_sec):
    print(f"動画数             : {s['n']} 本  (A {s['n_A']} / S {s['n_S']}, "
          f"総 {s['hours']*60:.1f}分)")
    print("・ 動画単位 (イベントが1件でも発生すれば衝突 — ① 従来判定と同一定義)")
    print(f"  混同行列         : TP {s['tp']} / FN {s['fn']} / FP {s['fp']} / TN {s['tn']}")
    print(f"  Recall {_fmt_pct(s['video_recall'])} / 誤検知率 {_fmt_pct(s['video_false_alarm'])} / "
          f"正解率 {_fmt_pct(s['accuracy'])} / 適合率 {_fmt_pct(s['precision'])} / "
          f"F1 {_fmt_pct(s['f1'])}")
    print("・ 位置基準 (サービス実用上の有効検出)")
    if s['n_A']:
        print(f"  Recall(開始 ±{tol_sec:g}秒) : {s['hit_recall']:.2f}%   "
              f"← ラベル開始近傍で開始したイベントのみを検出と認定")
        print(f"  参考: 区間重複のみ   : {s['overlap_recall']:.2f}%   "
              f"(クリップに衝突場面が含まれている場合まで認定)")
        print(f"  見逃し {s['missed']}件 / 誤位置検出 {s['wrong_place']}件 / "
              f"検出後分割 {s['split']}件")
    print(f"  誤検知イベント       : {s['false_events']}個"
          + (f"  (動画1時間あたり {s['false_per_hour']}個)" if s['false_per_hour'] is not None else "")
          + "   ← 衝突開始と無関係なクリップ全件")
    if s['err_median_f'] is not None:
        print(f"  開始時点の誤差       : 中央値 {s['err_median_f']:+d}f "
              f"({s['err_median_sec']:+.2f}秒)  ← 最も近いイベント基準")


def evaluate_service_path(
    model,
    folder_path,
    selected_files,
    target_id=config.TARGET_ID,
    r_value=config.R_VALUE,
    resize=config.RESIZE,
    clip_length=config.CLIP_LENGTH,
    base_rows=None,
):
    """サービスが実際に使用する predict_events_and_clips で評価する。

    上記の評価は粗い探索の選別・イベント状態遷移・結合/長さフィルタをスキップし「動画内に
    Aウィンドウが1つでもあるか」のみを評価(論文と同様の動画単位指標)。ここでは
    同一関数をクリップレンダリングのみ無効化して呼び出し、ユーザーが実際に受け取るイベントをラベルと
    照合する。

    判定基準 (本プロジェクトでは衝突の「開始」が重要であり終了は重要ではない):
      · 検出      : ラベル start_f ± EVAL_HIT_TOLERANCE_SEC 以内に開始したイベントが存在
      · 誤位置    : イベントは発生したが開始付近ではない → 検出とみなさない。
                    「イベントが1つでもあれば検出」とカウントすると recall が過大評価される。
      · 誤検知イベント: 衝突開始付近以外の全イベント(非衝突動画のイベント + 衝突
                    動画の余剰イベント)。ユーザーにとっての空振りクリップ数となる。
    """
    import tempfile
    from inference import predict_events_and_clips

    tol_sec = float(getattr(config, 'EVAL_HIT_TOLERANCE_SEC', 1.0))
    in_train = _training_video_names()

    print("\n" + "=" * 60)
    print("[② サービス判定]  サービスと同一関数: 粗探索 + 状態遷移 + マージ/最小長フィルタ")
    print("=" * 60)

    rows = []
    tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix='evalclips_'))
    for mp4_file in tqdm(selected_files, desc="サービス経路"):
        base = mp4_file.rsplit('.', 1)[0]
        video_path = os.path.join(folder_path, mp4_file)
        bboxes, gt_sf, gt_ef = _read_label(os.path.join(folder_path, f"{base}.txt"))
        if not bboxes:
            continue
        bbox = bboxes.get(target_id, next(iter(bboxes.values())))

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()

        stats = {}
        try:
            t0 = time.perf_counter()
            events = predict_events_and_clips(
                model, video_path=video_path, bbox=tuple(bbox),
                output_dir=tmp_dir, r_value=r_value, resize=resize,
                clip_length=clip_length, render_clips=False, stats=stats)
            proc_sec = time.perf_counter() - t0
        except Exception as exc:  # noqa: BLE001
            print(f"  [警告] {mp4_file} の処理に失敗: {exc}")
            continue

        frames_read = stats.get('frames', frames)
        base_r = (base_rows or {}).get(mp4_file, {})
        row = {'file': mp4_file, 'gt': 'A' if gt_sf is not None else 'S',
               # 2つの判定を動画別に並列表示 — ① 従来(ウィンドウ P(A)>0.5 1個↑) / ② サービス(イベント 1個↑)
               'base_pred': base_r.get('pred'), 'base_first_a': base_r.get('first_a'),
               'svc_pred': 'A' if events else 'S',
               'in_train': mp4_file in in_train, 'fps': round(fps, 2),
               # 夜間動画(config.NIGHT_VIDEOS, 目視確認済みリスト) — 光による誤検知を個別に確認するため
               'night': base in (getattr(config, 'NIGHT_VIDEOS', ()) or ()),
               'frames': frames, 'gt_start': gt_sf, 'gt_end': gt_ef,
               'n_events': len(events),
               'event_starts': ' '.join(str(e['start_frame']) for e in events),
               'hit': None, 'overlap': None, 'err_f': None, 'err_sec': None,
               'false_events': len(events),
               # ── 速度 (クリップ生成除外) ──
               'frames_read': frames_read,
               'proc_sec': round(proc_sec, 3),
               'proc_fps': round(frames_read / proc_sec, 2) if proc_sec else None,
               'realtime_x': round(frames_read / fps / proc_sec, 3) if proc_sec and fps else None,
               'decode_sec': round(stats.get('decode_sec', 0.0), 3),
               'flow_sec': round(stats.get('flow_sec', 0.0), 3),
               'infer_sec': round(stats.get('infer_sec', 0.0), 3),
               'windows_total': stats.get('windows_total', 0),
               'windows_eval': stats.get('windows_eval', 0)}
        if gt_sf is not None:
            tol = max(1, int(round(fps * tol_sec)))
            row['hit'] = False
            row['overlap'] = any(e['event_start_frame'] <= gt_ef
                                 and e['event_end_frame'] >= gt_sf for e in events)
            if events:
                best = min(events, key=lambda e: abs(e['start_frame'] - gt_sf))
                err = best['start_frame'] - gt_sf
                row['err_f'] = err
                row['err_sec'] = round(err / fps, 3)
                row['hit'] = abs(err) <= tol
                row['false_events'] = len(events) - (1 if row['hit'] else 0)
        rows.append(row)

    # ── 動画別結果一覧 ──
    print(f"\n{'動画':<22} {'学習フォルダ':>6} {'fps':>4} {'ラベル開始':>7} {'イベント':>5} "
          f"{'最寄の開始(誤差)':>24} {'処理fps':>7}  判定")
    print("-" * 100)
    for r in rows:
        if r['gt'] == 'A':
            if r['n_events'] == 0:
                verdict = "❌ 見逃し"
            elif r['hit']:
                verdict = "検出" + (f" +誤検知{r['false_events']}" if r['false_events'] else "")
            else:
                verdict = f"❌ 誤位置 ({r['n_events']}件)"
            near = (f"{r['gt_start'] + r['err_f']} ({r['err_f']:+d}f, {r['err_sec']:+.2f}s)"
                    if r['err_f'] is not None else "-")
        else:
            verdict = f"❌ 誤検知 {r['n_events']}件" if r['n_events'] else "正常"
            near = "-"
        print(f"{r['file']:<22} {'⚠あり' if r['in_train'] else '-':>6} {r['fps']:>4.0f} "
              f"{r['gt_start'] if r['gt_start'] is not None else '-':>7} {r['n_events']:>5} "
              f"{near:>24} {r['proc_fps'] or 0:>7.1f}  {verdict}")

    # ── 集計 ──
    print("\n" + "-" * 60)
    all_s = _summarize(rows)
    _print_summary(all_s, tol_sec)
    # 夜間動画のみ別途集計
    night = [r for r in rows if r.get('night')]
    night_line = None
    if night:
        na = [r for r in night if r['gt'] == 'A']
        ns = [r for r in night if r['gt'] == 'S']
        miss = [r['file'].rsplit('.', 1)[0][7:] for r in na if not r['overlap']]
        fa = [r['file'].rsplit('.', 1)[0][7:] for r in ns if r['n_events']]
        night_line = (f"動画 {len(night)}本 — 事故 {len(na)}本中 区間重複 {len(na) - len(miss)}本"
                      + (f"(見逃し {', '.join(miss)})" if miss else "")
                      + f" ・ 正常 {len(ns)}本中 誤検知 {len(fa)}本" + (f"({', '.join(fa)})" if fa else ""))
        print(f"\n[夜間動画(config.NIGHT_VIDEOS)] {night_line}")

    n_contam = sum(1 for r in rows if r['in_train'])
    clean_s = None
    if n_contam:
        clean = [r for r in rows if not r['in_train']]
        print(f"\n⚠️  評価動画 {len(rows)}本中 {n_contam}本が学習フォルダ({config.DATA_DIR})にも " \
              f"存在します。\n    それらの動画は学習に使用された可能性があり、上記の数値は見かけ上高くなっています。 " \
              f"学習フォルダに存在しない動画のみ別途集計:")
        print("-" * 60)
        if clean:
            clean_s = _summarize(clean)
            _print_summary(clean_s, tol_sec)
        else:
            print("  (該当なし)")
    print("=" * 60)

    out = {'hit_tol_sec': tol_sec,
           **{k: v for k, v in all_s.items() if k not in ('n', 'n_A', 'n_S', 'hours')},
           'n_in_train': n_contam, 'speed': _speed_summary(rows)}
    if clean_s:
        out.update({'clean_n': clean_s['n'], 'clean_svc_hit_recall': clean_s['hit_recall'],
                    'clean_svc_false_events': clean_s['false_events'],
                    'clean_svc_video_false_alarm': clean_s['video_false_alarm']})
    out['night'] = night_line
    out['rows'] = rows
    return out


class _Tee:
    """コンソール出力とファイル記録を同時に行う。"""

    def __init__(self, path):
        self.file = open(path, 'w', encoding='utf-8')
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def _info_metrics(m, prefix, rec_key, fa_key):
    """run_info 1行出力: 再現率・誤検知率・正解率・適合率・F1 + 混同行列。"""
    g = lambda k: m.get(prefix + k)  # noqa: E731
    return (f"再現率 {_fmt_pct(g(rec_key))} / 誤検知率 {_fmt_pct(g(fa_key))} / "
            f"正解率 {_fmt_pct(g('accuracy'))} / 適合率 {_fmt_pct(g('precision'))} / "
            f"F1 {_fmt_pct(g('f1'))} (TP {g('tp')} FN {g('fn')} FP {g('fp')} TN {g('tn')})")


def evaluate_folder_accuracy(model, folder_path=config.EVAL_FOLDER_PATH, **kwargs):
    """フォルダ評価のエントリポイント — 結果をコンソールとファイルに同時に保存する。

    保存内容:
      1) 実行フォルダ       outputs/evallogs/<YYMMDD_HHMM>_<モデル>_<ptY|ptN>[_span]/
           · コンソール全文ログ  eval_<モデル>_<ptY|ptN>_<日時>.txt
           · 動画別結果     eval_<モデル>_<ptY|ptN>_<日時>_videos.csv
      2) 要約1行      outputs/evallogs/eval_summary.csv  (追記、run 列 = 実行フォルダ名)
         → 5種類の学習組み合わせを1つの表で比較するために継続追記する。
    """
    log_dir = pathlib.Path(getattr(config, 'EVAL_LOG_DIR',
                                   pathlib.Path('outputs') / 'evallogs'))
    log_dir.mkdir(parents=True, exist_ok=True)
    weights = pathlib.Path(str(config.EVAL_WEIGHTS_PATH)).name
    # 事前学習の有無は評価対象の重みファイル名(_ptY_/_ptN_)から判定。名前にない場合はconfig値。
    tag = next((t for t in ('ptY', 'ptN') if f'_{t}_' in weights),
               getattr(config, 'PRETRAIN_TAG', '?'))
    # Aクリップ方式 span で学習した重みの場合は実行フォルダ・ファイル名に _span を付与し 'pre' 結果と区別する。
    # バッチ構成(--balance)で学習した重みは _bal、低画質増強(TRAIN_LQ_AUG_PROB)は _lq も付与。
    # fps増強(TRAIN_FPS_AUG_PROB)は _fps、224以外の入力サイズ(INPUT_SIZE)は _s<サイズ>、ヘッド時間プーリング(HEAD_TEMPORAL_POOL)は _top3 等。
    name_tag = (tag + ('_span' if '_span_' in weights else '')
                + ('_bal' if '_bal_' in weights else '')
                + ('_lq' if '_lq_' in weights else '_lqw' if '_lqw_' in weights else '')
                + ('_fps' if '_fps_' in weights else '')
                + ('_light' if '_light_' in weights else '')
                + (f'_s{config.RESIZE[0]}' if config.RESIZE[0] != 224 else '')
                + (f"_{config.HEAD_TEMPORAL_POOL}" if getattr(config, 'HEAD_TEMPORAL_POOL', 'avg') != 'avg' else ''))
    # クロップ r を変更して評価した場合(main --r)は名前に _r<値> を付与 — 要約表の run 列で区別
    r_value = float(kwargs.get('r_value', config.R_VALUE))
    if abs(r_value - 1.0) > 1e-9:
        name_tag += f'_r{r_value:g}'
    run_dir, stamp = make_run_dir(log_dir, config.MODEL_NAME, name_tag)
    log_path = run_dir / f"eval_{config.MODEL_NAME}_{name_tag}_{stamp}.txt"
    write_run_info(run_dir, [
        ('種類', '評価'),
        ('モデル', config.MODEL_NAME),
        ('事前学習', tag),
        ('開始', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
        ('評価重み', str(config.EVAL_WEIGHTS_PATH)),
        ('評価フォルダ', str(folder_path)),
        ('10fps未満動画の除外(config.LOW_FPS_VIDEOS)', '有効' if getattr(config, 'EXCLUDE_VIDEOS', None) else '無効'),
        ('クロップ r', f'{r_value:g}'),
        ('後処理fps適合(粗い探索間隔・最小長さ)',
         '有効' if config.PREDICT_FPS_SCALED_POSTPROC else '無効'),
        # 速度指標は本設定が同一の評価間でのみ比較する(2026-10-22以降デフォルト無効 — GPU推論 約2.6倍高速)。
        # 精度指標は有効/無効の差が P(A) 1e-6 レベルであるためそのまま比較可能。
        ('GPUメモリ配置 channels_last',
         '有効' if (is_channels_last_3d_supported(next(model.parameters()).device)
                 and config.USE_CHANNELS_LAST) else '無効'),
    ])

    tee = _Tee(log_path)
    sys.stdout = tee
    try:
        print(f"[評価対象] モデル={config.MODEL_NAME} / 事前学習={tag} / 重み={weights} / クロップ r={r_value:g}")
        print(f"[評価フォルダ] {folder_path}")
        m = _evaluate_folder_impl(model, folder_path=folder_path, **kwargs)
    finally:
        sys.stdout = tee.stdout
        tee.close()

    if not m:
        print(f"[ログ] 評価ログ: {log_path}")
        return m

    speed = m.pop('_speed', None) or {}
    s = speed.get('service') or {}
    info = (pathlib.Path(run_dir, 'run_info.txt').read_text(encoding='utf-8')
            .rstrip('\n').split('\n'))
    write_run_info(run_dir, [tuple(l.split(': ', 1)) for l in info] + [
        ('終了', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
        ('評価動画数', m.get('n_videos')),
        ('① 従来判定(ウィンドウ P(A)>0.5 1個↑)', _info_metrics(m, 'base_', 'recall', 'false_alarm')),
        ('② サービス判定(イベント 1個↑)', _info_metrics(m, 'svc_', 'video_recall', 'video_false_alarm')),
        (f"② サービス位置基準再現率(開始 ±{m.get('svc_hit_tol_sec')}秒)", _fmt_pct(m.get('svc_hit_recall'))),
        ('② サービス誤検知イベント数', m.get('svc_false_events')),
        ('学習フォルダ重複動画数', m.get('n_in_train')),
        ('夜間動画(config.NIGHT_VIDEOS)', m.get('svc_night') or 'なし'),
        ('推論デバイス', speed.get('device')),
        ('処理速度(サービス経路)', f"秒あたり {s['proc_fps']:.1f} フレーム, 実時間の {s['realtime_x']:.2f}倍速 "
                            f"(クリップ生成除外)" if s else None),
        ('モデル推論速度', ' / '.join(f"バッチ {b['batch']}: 秒あたり {b['clips_per_sec']:.1f}個の窓"
                                      for b in speed.get('model', [])) or None),
    ])

    # 動画別結果 CSV 出力
    rows = m.pop('_svc_rows', [])
    if rows:
        per_video = run_dir / f"eval_{config.MODEL_NAME}_{name_tag}_{stamp}_videos.csv"
        with open(per_video, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[ログ] 動画別結果: {per_video}")

    # サマリ CSV に1行追記 (モデル比較用)
    summary = log_dir / 'eval_summary.csv'
    # base_* = ① 従来判定(ウィンドウ P(A)>0.5 1個↑), svc_* = ② サービス判定(イベント 1個↑)。
    # 両者の tp~f1 は同一定義であるためそのまま比較可能。svc_hit_recall は位置基準(開始 ±許容誤差)。
    # ⚠️ 2026-10-12 以前の要約ファイルの 'recall' は現在の base_recall、'svc_recall' は
    #    現在の svc_hit_recall (名称のみ変更)。
    cols = ['datetime', 'run', 'model', 'pretrain', 'weights', 'eval_folder', 'n_videos',
            'base_tp', 'base_fn', 'base_fp', 'base_tn', 'base_recall', 'base_false_alarm',
            'base_accuracy', 'base_precision', 'base_f1',
            'svc_tp', 'svc_fn', 'svc_fp', 'svc_tn', 'svc_video_recall', 'svc_video_false_alarm',
            'svc_accuracy', 'svc_precision', 'svc_f1',
            'svc_hit_tol_sec', 'svc_hit_recall', 'svc_overlap_recall', 'svc_missed',
            'svc_wrong_place', 'svc_split', 'svc_false_events', 'svc_false_per_hour',
            'svc_err_median_f', 'svc_err_median_sec',
            'n_in_train', 'clean_n', 'clean_svc_hit_recall', 'clean_svc_false_events',
            'clean_svc_video_false_alarm',
            # 推論速度 — デバイスが異なる場合は比較できないため device と併せて確認
            'device', 'speed_proc_fps', 'speed_realtime_x', 'speed_windows_eval_pct',
            'speed_model_batch', 'speed_model_clips_per_sec', 'speed_model_ms_per_clip']
    # カラム構成が変更された古い要約ファイルに追記すると列がずれるため退避する
    if summary.exists():
        with open(summary, encoding='utf-8') as f:
            old_header = f.readline().strip().split(',')
        if old_header != cols:
            moved = summary.with_name(f"eval_summary_old_{stamp}.csv")
            summary.rename(moved)
            print(f"[ログ] サマリ CSV のカラムが変更されたため既存ファイルを {moved.name} に退避しました")
    row = {'datetime': datetime.now().strftime('%Y-%m-%d %H:%M'), 'run': run_dir.name,
           'model': config.MODEL_NAME, 'pretrain': tag, 'weights': weights,
           'eval_folder': str(folder_path), **m}
    write_header = not summary.exists()
    with open(summary, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(cols)
        w.writerow(['' if row.get(c) is None else row.get(c, '') for c in cols])

    print(f"[ログ] 評価全文: {log_path}")
    print(f"[ログ] サマリ累積: {summary}")
    return m
