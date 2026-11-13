import os

import torch
import torch.nn as nn
from pytorchvideo.models.hub import x3d_m
from torch.utils.checkpoint import checkpoint, checkpoint_sequential


def _tag_names(path):
    """条件表記を探索する名称群 — ファイル名、および学習セッションフォルダ(<日時>_<モデル>_<条件>)内のファイルはそのフォルダ名。
    work_best.pth・work_cand_ep*.pth・epochs/ 内ファイルのようにファイル名に表記がなくともフォルダ名で照合。"""
    import re
    p = os.path.abspath(str(path))
    names = [os.path.basename(p)]
    for d in (os.path.dirname(p), os.path.dirname(os.path.dirname(p))):
        if re.match(r'^\d{6}_\d{4}', os.path.basename(d)):
            names.append(os.path.basename(d))
            break
    return names


def weights_input_size(path):
    """重みファイル(またはセッションフォルダ)名に含まれる入力サイズ表記(_s256等 — 224以外のサイズで学習時に train.py が付与)。未検出時は None。"""
    import re
    for n in _tag_names(path):
        m = re.search(r'_s(\d{3})(?=[_.]|$)', n)
        if m:
            return int(m.group(1))
    return None


def check_input_size(path):
    """重みの学習時入力サイズと現在の config.INPUT_SIZE が異なる場合に実行を停止 — サイズ不一致時は判定が歪むため。
    ファイル名に表記がない場合は 224 で学習されたものとして扱う。"""
    import config
    n = weights_input_size(path) or 224
    if n != config.RESIZE[0]:
        raise RuntimeError(f"重み {os.path.basename(str(path))} は入力 {n}×{n} で学習されていますが、現在の入力サイズは "
                           f"{config.RESIZE[0]} です — 環境変数 INPUT_SIZE={n} で実行してください(config.INPUT_SIZE)")


HEAD_POOL_MODES = ('avg', 'top3', 'avgtop3')


def weights_head_pool(path):
    """重みファイル(またはセッションフォルダ)名に含まれるヘッド時間プーリング表記(_top3・_avgtop3 — avg以外のプーリングで学習時に付与)。未検出時は 'avg'。"""
    import re
    for n in _tag_names(path):
        m = re.search(r'_(avgtop3|top3)(?=[_.]|$)', n)
        if m:
            return m.group(1)
    return 'avg'


def check_head_pool(path):
    """重みの学習時ヘッド時間プーリングと現在の config.HEAD_TEMPORAL_POOL が異なる場合に実行を停止。
    プーリング層には学習パラメータが存在しないため異なる設定でも読込自体は成功してしまい判定のみが狂うため、名称で厳密に防ぐ。"""
    import config
    mode = weights_head_pool(path)
    if mode != config.HEAD_TEMPORAL_POOL:
        raise RuntimeError(f"重み {os.path.basename(str(path))} はヘッド時間プーリング {mode} で学習されていますが、現在は "
                           f"{config.HEAD_TEMPORAL_POOL} です — 環境変数 HEAD_TEMPORAL_POOL={mode} で実行してください"
                           f"(config.HEAD_TEMPORAL_POOL)")


def check_weights(path):
    """重み読込前の学習条件検証 — 入力サイズ(_s256)・ヘッド時間プーリング(_top3等)が現在設定と一致すること。"""
    check_input_size(path)
    check_head_pool(path)


class TemporalPool(nn.Module):
    """ヘッドの時間軸集約 (B, C, T, H, W) → (B, C, 1, H, W) — config.HEAD_TEMPORAL_POOL。

    top3   : チャネル・位置ごとに時間軸で最大値上位k(=3)個を平均。2〜6フレームの瞬間的な揺れが1/30に希釈されるのを防止。
    avgtop3: 全体平均と上位k個平均の平均 — 長期コンテキスト(平均)と瞬間特徴(上位k)を併用。事前学習の
             post_conv 入力ダイナミックレンジから大きく逸脱しない利点がある。
    avg は本モジュールではなく従来の AdaptiveAvgPool3d を維持（過去学習モデルとのビット完全互換を担保）。
    mean・topk の標準演算のみ使用しているため ONNX(TopK)・OpenVINO への変換も完全互換（infer_runtime 自己診断で検証済み）。
    """

    def __init__(self, mode, k=3):
        super().__init__()
        if mode not in ('top3', 'avgtop3'):
            raise ValueError(f'TemporalPool は top3・avgtop3 のみ対応: {mode!r}')
        self.mode, self.k = mode, int(k)

    def forward(self, x):
        top = torch.topk(x, min(self.k, x.shape[2]), dim=2).values.mean(dim=2, keepdim=True)
        if self.mode == 'top3':
            return top
        return 0.5 * (x.mean(dim=2, keepdim=True) + top)


# X3D-M 最終 res ステージの出力チャネル数（構造定義に基づく検証値）。
X3D_M_FEATURE_CHANNELS = 192
# ヘッド拡張部(post_conv)通過後のチャネル数 — 分類およびCAMはこの次元で算出される。
X3D_HEAD_CHANNELS = 2048


# 分類ヘッド直前の dropout 確率 — 3種のバックボーン共通設定値。
# バックボーン比較実験において正則化強度がモデルごとに異なると結果の公平な解釈が不可能となるため、
# 厳密に同一値を適用。（従来: s3d 0.2 / x3d 0.5 / slowfast 0.2 と不揃いであった）
DROPOUT_P = 0.5

# gradient checkpointing — 順伝播の中間活性化値を保持せず逆伝播時に再計算。
# 勾配は数学的に完全一致するため **認識精度への影響はゼロ** であり、GPUメモリ消費を削減する代わりに
# 計算時間が約30%増加する。
#
# X3D はパラメータ数が S3D の 1/2.7(2.98M vs 7.91M)である一方、活性化値メモリは1.7倍に達する(25.4GB vs
# 15.0GB、バッチ15基準)。時間軸30フレームを縮約せずボトルネック内でチャネルを2.25倍に拡張するためである。
# RTX 3080 10GB 環境でバッチ15が OOM となる主因であり、バッチサイズ削減（→ バッチ内A不在による学習崩壊）の代償として本機能を適用。
# 学習時のみ機能し（self.training）、推論・CAM 経路には一切関与しない。
# 学習時のみ機能し(self.training)、推論・CAM経路には関与しない。
GRAD_CHECKPOINT = os.getenv("GRAD_CHECKPOINT", "1").strip().lower() not in ("0", "false", "no")
# バックボーンブロックが5個のみのため、segments=5 ではセグメントあたり1ブロックとなり、ブロック丸ごと再計算が必要。
# features[1] でのピークメモリが 8.67GB(バッチ15) となり 10GB 上限に極めて逼迫する。
# ブロック数より細かく分割することで checkpoint_sequential が内部まで分割しピークメモリを抑制可能。
GRAD_CHECKPOINT_SEGMENTS = int(os.getenv("GRAD_CHECKPOINT_SEGMENTS", "5"))
# チェックポインティング対象とする前方ブロック数（features の stem・res2・res3・res4・res5 のうち先頭からの件数）。空の場合は
# checkpoint_sequential（セグメント5 → stem〜res4 を逆伝播時に再計算、res5 は非チェックポイント = 4 と等価）。
# メモリに余裕がある場合は件数を減らし再計算負荷を軽減可能 — python -m model.speed_check ⑤ で時間・メモリを測定して選定。
# ⚠️ チェックポイント対象ブロックは再計算時に BN 移動平均（推論用）も同一バッチ統計で再度更新される
#    （2026-10-28 確認 — momentum 0.1 を2回 = 0.19 を1回。勾配は完全一致）。過去の学習がその挙動であったため、
#    適用範囲を変更したブロックは BN momentum を補正し従来と同一の移動平均を維持（_match_bn_momentum）。
_ckpt_blocks_env = os.getenv("GRAD_CHECKPOINT_BLOCKS", "").strip()
GRAD_CHECKPOINT_BLOCKS = int(_ckpt_blocks_env) if _ckpt_blocks_env else None

class HitAndRun3DCNN(nn.Module):
    """X3D-M バックボーンベースの当て逃げ・物損事故検知モデル。

    X3D(Feichtenhofer, CVPR 2020)は2D画像分類ネットワークを幅・深さ・解像度・フレーム軸の
    4次元で漸進拡張した高効率3D-CNNである。同一精度をS3Dより大幅に少ない演算量で達成するのが特徴であり、
    マルチアワー動画をスライディングウィンドウで走査する本サービスにおいて「ウィンドウあたりの推論コスト」を
    低減する上で極めて高い優位性を持つ。
    (パラメータ数: X3D-M 3.8M vs S3D 7.9M)

    ⚠️ pytorchvideo のデフォルトヘッド(ResNetBasicHead)は AvgPool3d カーネルが (16,7,7) で固定されており、
       T=30 入力でクラッシュする。しかしヘッド全体を破棄すると内部の拡張 conv(192→432→2048) も失われ、
       事前学習済みの重みを移植できなくなる。
       そのため **固定カーネル pool のみを** 置換する:

           AvgPool3d((16,7,7))  →  AdaptiveAvgPool3d((1, None, None))

       時間軸のみ集約し空間解像度 7×7 を保持するため、(a) 任意のフレーム長 T に対応可能かつ
       (b) 2048チャネルの空間特徴マップが温存され CAM の抽出が可能となる。
       CAM は空間特徴の時間平均で算出されるため(inference.py の mean(cam, dim=0))、時間軸の保持は
       不要であり、post_conv を T=1 で実行することで演算量も削減される。

    サービス推論（inference.py）が使用する公開インターフェース:
      - 入力 (B, 3, T, 224, 224) → 出力 (B, num_classes) logits
      - `head_conv`   : 1×1×1 Conv3d 分類ヘッド（CAM 重みとして使用）
      - `cam_layer`   : CAM forward hook 対象層（2048チャネル空間特徴マップ）
      - `forward_with_cam` : logits + 位置別クラスマップ(CAM) — 事故クリップヒートマップ生成用

    Args:
        num_classes: 分類クラス数 (デフォルト 2: S/A)
        pretrained : True の場合 Kinetics-400 事前学習重みで初期化。
                     推論時は学習済み state_dict をロードするため False でインスタンス化し
                     不要なダウンロードを回避する。
    """

    def __init__(self, num_classes=2, pretrained=False, dropout_p=DROPOUT_P, temporal_pool=None):
        super(HitAndRun3DCNN, self).__init__()
        base = x3d_m(pretrained=pretrained)
        # ヘッド時間プーリング(config.HEAD_TEMPORAL_POOL) — avg は標準演算、top3・avgtop3 は TemporalPool。重みキーは完全互換。
        if temporal_pool is None:
            import config
            temporal_pool = getattr(config, 'HEAD_TEMPORAL_POOL', 'avg')
        if temporal_pool not in HEAD_POOL_MODES:
            raise ValueError(f"HEAD_TEMPORAL_POOL は {'・'.join(HEAD_POOL_MODES)} のいずれかである必要があります: {temporal_pool!r}")
        self.temporal_pool = temporal_pool

        # blocks[-1] は分類ヘッド。stem + res ステージと分離する。
        self.features = nn.Sequential(*list(base.blocks)[:-1])

        # ヘッドの拡張部(pre_conv 192→432 → BN → ReLU → pool → post_conv 432→2048
        # → ReLU) は維持し、固定カーネル pool のみ置換する。
        head_pool = base.blocks[-1].pool
        if temporal_pool == 'avg':
            head_pool.pool = nn.AdaptiveAvgPool3d((1, None, None))
        else:
            import config
            head_pool.pool = TemporalPool(temporal_pool, getattr(config, 'HEAD_TEMPORAL_TOPK', 3))
        self.head_pool = head_pool

        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=dropout_p)
        # 分類ヘッドは logit 出力のため BN/ReLU を持たない Conv3d 単独構造（CAM 重み算出と兼用）。
        # 元ヘッドの Linear(2048→C) と数学的に等価な演算のため重みを reshape して完全移植可能。
        self.head_conv = nn.Conv3d(
            X3D_HEAD_CHANNELS, num_classes, kernel_size=1)
        self._match_bn_momentum()

    def default_checkpoint_blocks(self):
        """GRAD_CHECKPOINT_BLOCKS 未指定時(checkpoint_sequential)に再計算対象とする前方ブロック数 — デフォルト 4。"""
        n = len(self.features)
        seg = max(1, min(GRAD_CHECKPOINT_SEGMENTS, n))
        return (n // seg) * (seg - 1)

    def _match_bn_momentum(self):
        """チェックポイント範囲がデフォルトと異なるブロックの BN momentum を補正（上記 GRAD_CHECKPOINT_BLOCKS コメント参照）。

        同一バッチ統計で momentum m を2回適用 = 1-(1-m)² を1回適用と等価。デフォルトで再計算していたブロックが除外された場合は
        数値を引き上げ、新規に追加されたブロックは 1-√(1-m) に引き下げる。momentum は重みファイルに含まれず推論結果には影響しない。
        """
        if not GRAD_CHECKPOINT or GRAD_CHECKPOINT_BLOCKS is None:
            return
        k, d = GRAD_CHECKPOINT_BLOCKS, self.default_checkpoint_blocks()
        for i in range(min(k, d), min(max(k, d), len(self.features))):
            for m in self.features[i].modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.momentum is not None:
                    m.momentum = (1 - (1 - m.momentum) ** 2 if i >= k
                                  else 1 - (1 - m.momentum) ** 0.5)

    @property
    def cam_layer(self):
        """CAM 用特徴マップを取り出す層 — ヘッド拡張部出力(2048チャネル、空間 7×7 を保持)。

        property 定義のためモジュールが重複登録されない（重みキーに影響なし）。inference.py が
        `model.cam_layer.register_forward_hook(...)` で特徴マップを取得。
        head_conv.weight(2048チャネル) とチャネル数が一致することで CAM の線形結合が成立する。
        （旧 S3D バックボーンの層名 inception5b を踏襲していたものを役割に応じてリネーム）
        """
        return self.head_pool

    def forward(self, x):
        if self.training and GRAD_CHECKPOINT and GRAD_CHECKPOINT_BLOCKS is not None:
            # 前方 GRAD_CHECKPOINT_BLOCKS 個のみブロック単位でチェックポイント(checkpoint_sequential と等価の計算)
            for i, block in enumerate(self.features):
                x = (checkpoint(block, x, use_reentrant=False)
                     if i < GRAD_CHECKPOINT_BLOCKS else block(x))
        elif self.training and GRAD_CHECKPOINT:
            # バックボーンのみチェックポイント。head_pool は CAM hook 対象のため除外。
            x = checkpoint_sequential(
                self.features, GRAD_CHECKPOINT_SEGMENTS, x, use_reentrant=False)
        else:
            x = self.features(x)    # (B, 192, T, 7, 7)
        x = self.head_pool(x)       # (B, 2048, 1, 7, 7)
        x = self.avg_pool(x)        # (B, 2048, 1, 1, 1)
        x = self.dropout(x)
        x = self.head_conv(x)
        return x.squeeze(-1).squeeze(-1).squeeze(-1)

    def forward_with_cam(self, x):
        """forward と同一の logits (B, C) + 位置別クラスマップ(CAM) (B, C, 7, 7) — 推論専用。

        CAM は分類ヘッド(1×1×1 Conv) を空間平均の「前に」 7×7 の各グリッドに適用したもの。空間平均と 1×1 Conv は
        演算順序を入れ替えても結果が一致するため、グリッド平均がそのまま logits となる。グリッドごとに 'A − S' を見ることで
        その領域が事故判定にどれだけ寄与したかのヒートマップとなる。判定時に算出済みの特徴マップを流用するため、
        追加コストは49グリッドの 1×1 乗算のみであり、高速推論エンジン(ONNX・OpenVINO)でも同一パスで同時出力される(infer_runtime.py)。
        logits は forward と同一経路(空間平均 → Conv)で計算され判定結果の数学的一致を保証する。
        """
        x = self.head_pool(self.features(x))            # (B, 2048, 1, 7, 7)
        cam = self.head_conv(x)[:, :, 0]                # (B, C, 7, 7)
        x = self.head_conv(self.dropout(self.avg_pool(x)))
        return x.squeeze(-1).squeeze(-1).squeeze(-1), cam
