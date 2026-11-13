import os

import torch
import torch.nn as nn
from pytorchvideo.models.hub import x3d_m
from torch.utils.checkpoint import checkpoint, checkpoint_sequential


def _tag_names(path):
    """条件タグを探索する名前のリスト — ファイル名、および学習実行回フォルダ (<日時>_<モデル>_<条件>) 配下のファイルであればそのフォルダ名。
    work_best.pth・work_cand_ep*.pth・epochs/ 内のファイルのようにファイル名にタグがなくても実行回フォルダ名から取得可能。"""
    import re
    p = os.path.abspath(str(path))
    names = [os.path.basename(p)]
    for d in (os.path.dirname(p), os.path.dirname(os.path.dirname(p))):
        if re.match(r'^\d{6}_\d{4}', os.path.basename(d)):
            names.append(os.path.basename(d))
            break
    return names


def weights_input_size(path):
    """重みファイル (または実行回フォルダ) 名に含まれる入力サイズタグ (_s256 等 — 224 以外のサイズで学習時に train.py が付与)。なければ None。"""
    import re
    for n in _tag_names(path):
        m = re.search(r'_s(\d{3})(?=[_.]|$)', n)
        if m:
            return int(m.group(1))
    return None


def check_input_size(path):
    """重みの学習時入力サイズと現在の config.INPUT_SIZE が異なる場合に停止する — サイズが異なると判定が崩れるため。
    ファイル名にタグがない場合は 224 で学習されたものとみなす。"""
    import config
    n = weights_input_size(path) or 224
    if n != config.RESIZE[0]:
        raise RuntimeError(f"重み {os.path.basename(str(path))} は入力 {n}×{n} で学習されていますが、現在の入力サイズは "
                           f"{config.RESIZE[0]} です — 環境変数 INPUT_SIZE={n} で実行してください (config.INPUT_SIZE)")


HEAD_POOL_MODES = ('avg', 'top3', 'avgtop3')


def weights_head_pool(path):
    """重みファイル (または実行回フォルダ) 名に含まれるヘッド時間プーリングタグ (_top3・_avgtop3 — avg 以外で学習時に train.py が付与)。なければ 'avg'。"""
    import re
    for n in _tag_names(path):
        m = re.search(r'_(avgtop3|top3)(?=[_.]|$)', n)
        if m:
            return m.group(1)
    return 'avg'


def check_head_pool(path):
    """重みの学習時ヘッド時間プーリングと現在の config.HEAD_TEMPORAL_POOL が異なる場合に停止する。
    プーリング層には学習パラメータがないため異なる設定でも重み自体は読み込めてしまい、判定のみが狂うためファイル名で防止する。"""
    import config
    mode = weights_head_pool(path)
    if mode != config.HEAD_TEMPORAL_POOL:
        raise RuntimeError(f"重み {os.path.basename(str(path))} はヘッド時間プーリング {mode} で学習されていますが、現在は "
                           f"{config.HEAD_TEMPORAL_POOL} です — 環境変数 HEAD_TEMPORAL_POOL={mode} で実行してください"
                           f" (config.HEAD_TEMPORAL_POOL)")


def check_weights(path):
    """重み読み込み前の学習条件チェック — 入力サイズ (_s256)・ヘッド時間プーリング (_top3 等) が現在の設定と一致している必要がある。"""
    check_input_size(path)
    check_head_pool(path)


class TemporalPool(nn.Module):
    """ヘッドの時間軸プーリング (B, C, T, H, W) → (B, C, 1, H, W) — config.HEAD_TEMPORAL_POOL。

    top3   : チャネル・グリッドごとに時間軸で最大の上位 k(=3) 個を平均。2〜6フレーム程度の短い揺れが 1/30 に希釈されない。
    avgtop3: 全体平均と上位 k 個平均の平均 — 長期コンテキスト (平均) と一瞬のイベント (上位 k) を両立。値域が中間となるため
             事前学習モデルの post_conv 入力範囲から大きく逸脱しない。
    avg は本モジュールを使用せず元の AdaptiveAvgPool3d を維持 (従来の学習結果とビット完全一致を担保)。
    mean・topk のみを使用しているため ONNX(TopK)・OpenVINO への変換も完全互換 (infer_runtime の自己診断で検証済み)。
    """

    def __init__(self, mode, k=3):
        super().__init__()
        if mode not in ('top3', 'avgtop3'):
            raise ValueError(f'TemporalPool は top3・avgtop3 のみサポート: {mode!r}')
        self.mode, self.k = mode, int(k)

    def forward(self, x):
        top = torch.topk(x, min(self.k, x.shape[2]), dim=2).values.mean(dim=2, keepdim=True)
        if self.mode == 'top3':
            return top
        return 0.5 * (x.mean(dim=2, keepdim=True) + top)


# X3D-M の最終 res ステージの出力チャネル数 (アーキテクチャ確認により検証済み)。
X3D_M_FEATURE_CHANNELS = 192
# ヘッド拡張部 (post_conv) 通過後のチャネル数 — 分類/CAM はこの次元で計算される。
X3D_HEAD_CHANNELS = 2048


# 分類ヘッド前の Dropout 比率 — 全バックボーン共通値。
# バックボーン比較実験において正規化の強度がモデルごとに異なると結果の解釈が困難になるため、
# 必ず同一の値を使用する。(従来: s3d 0.2 / x3d 0.5 / slowfast 0.2 と不揃いであった)
DROPOUT_P = 0.5

# Gradient Checkpointing — 中間活性化を保持せず、backward 時に forward を再計算する。
# 勾配は数学的に完全に同一であるため **精度への悪影響はなく**、メモリ消費を大幅に削減し、
# 速度は約 30% 低下する。
#
# X3D はパラメータ数が S3D の 1/2.7 (2.98M vs 7.91M) であるのに対し、活性化メモリは 1.7倍 (25.4GB vs
# 15.0GB, batch 15 基準)。時間軸を 30 フレームそのまま維持しボトルネック内でチャネルを
# 2.25倍に拡張するためである。RTX 3080 10GB で batch 15 が OOM となる原因であるため、
# バッチサイズを減らす (→ バッチ内に A が含まれなくなり学習が破綻する) 代わりに本機能を使用する。
# 学習時のみ動作し (self.training)、推論・CAM 経路には影響しない。
GRAD_CHECKPOINT = os.getenv("GRAD_CHECKPOINT", "1").strip().lower() not in ("0", "false", "no")
# バックボーンブロックが5個のみのため、segments=5 の場合 1セグメントあたり1ブロック = ブロック全体を
# 再計算することになる。そのピークメモリは features[1] で 8.67GB (batch 15) となり 10GB に対して逼迫する。
# ブロック数より大きく設定することで checkpoint_sequential がブロック内部まで分割しピークを抑制できる。
GRAD_CHECKPOINT_SEGMENTS = int(os.getenv("GRAD_CHECKPOINT_SEGMENTS", "5"))
# チェックポインティング対象とする前方ブロック数 (features の stem・res2・res3・res4・res5 のうち前方何個か)。
# 未指定時は従来通りの checkpoint_sequential (セグメント 5 → stem〜res4 を backward 時に再計算、res5 は通常保持 = 4 と等価)。
# メモリに余裕がある場合は削減して再計算コストを抑制可能 — python -m model.speed_check ⑤ で RTX 3080 の時間・メモリを測定して選定。
# ⚠️ チェックポインティング対象ブロックは再計算時に BN 移動平均 (評価時に使用する統計量) も同一バッチ統計で再度更新される
#    (2026-10-28 検証 — momentum 0.1 を2回 = 0.19 を1回。勾配は同一)。従来の学習がその挙動であったため、
#    範囲が変更されるブロックは BN momentum を調整して従来と同様の移動平均を再現する (_match_bn_momentum)。
_ckpt_blocks_env = os.getenv("GRAD_CHECKPOINT_BLOCKS", "").strip()
GRAD_CHECKPOINT_BLOCKS = int(_ckpt_blocks_env) if _ckpt_blocks_env else None

class HitAndRun3DCNN(nn.Module):
    """X3D-M バックボーンベースの当て逃げ (物損事故) 検知モデル。

    X3D (Feichtenhofer, CVPR 2020) は 2D 画像分類ネットワークを幅・深さ・解像度・フレーム軸で
    段階的に拡張して得られた高効率 3D-CNN である。S3D と同等の精度を遥かに少ない演算量で達成するのが
    特徴であり、数時間の長時間動画をスライディングウィンドウで走査する本サービスにおいて、
    「ウィンドウあたりの処理コスト」を削減する大きな利点がある。
    (パラメータ数: X3D-M 3.8M vs S3D 7.9M)

    ⚠️ pytorchvideo のデフォルトヘッド (ResNetBasicHead) は AvgPool3d カーネルが (16,7,7) に
       固定されており、T=30 の入力で形状不一致エラーとなる。しかしヘッド全体を破棄すると内部の
       拡張 conv (192→432→2048) も同時に失われ、元の構造で学習された重みを移植できなくなる。
       そのため、**固定カーネルの pool のみ** を差し替える:

           AvgPool3d((16,7,7))  →  AdaptiveAvgPool3d((1, None, None))

       時間軸のみを集約し空間 7×7 を保持するため、(a) 任意の T で動作し、
       (b) 2048チャネルの空間特徴マップが残存するため CAM を抽出できる。
       CAM は最終的に時間軸で平均化されるため (inference.py の mean(cam, dim=0))、
       時間軸の保持は不要であり、post_conv を T=1 のみで実行することで演算量も削減できる。

    本番サービス (inference.py) が利用するインターフェース:
      - 入力 (B, 3, T, 224, 224) → 出力 (B, num_classes) logits
      - `head_conv`   : 1×1×1 Conv3d 分類ヘッド (CAM 重みとして使用)
      - `cam_layer`   : CAM forward hook 対象 (2048チャネル空間特徴マップ)
      - `forward_with_cam` : logits + グリッド別クラス活性化マップ (CAM) — サービス事故クリップヒートマップ用

    Args:
        num_classes: 分類クラス数 (デフォルト 2: S/A)
        pretrained : True の場合、Kinetics-400 事前学習重みでバックボーンを初期化。
                     推論時は学習済み state_dict をロードするため False で生成し不要なダウンロードを回避。
    """

    def __init__(self, num_classes=2, pretrained=False, dropout_p=DROPOUT_P, temporal_pool=None):
        super(HitAndRun3DCNN, self).__init__()
        base = x3d_m(pretrained=pretrained)
        # ヘッド時間プーリング (config.HEAD_TEMPORAL_POOL) — avg は元の演算、top3・avgtop3 は TemporalPool。重みキーは同一。
        if temporal_pool is None:
            import config
            temporal_pool = getattr(config, 'HEAD_TEMPORAL_POOL', 'avg')
        if temporal_pool not in HEAD_POOL_MODES:
            raise ValueError(f"HEAD_TEMPORAL_POOL は {'・'.join(HEAD_POOL_MODES)} のいずれかである必要があります: {temporal_pool!r}")
        self.temporal_pool = temporal_pool

        # blocks[-1] はヘッド。stem+res ステージと分離する。
        self.features = nn.Sequential(*list(base.blocks)[:-1])

        # ヘッドの拡張部 (pre_conv 192→432 → BN → ReLU → pool → post_conv 432→2048
        # → ReLU) は維持し、固定カーネル pool のみを差し替える。
        head_pool = base.blocks[-1].pool
        if temporal_pool == 'avg':
            head_pool.pool = nn.AdaptiveAvgPool3d((1, None, None))
        else:
            import config
            head_pool.pool = TemporalPool(temporal_pool, getattr(config, 'HEAD_TEMPORAL_TOPK', 3))
        self.head_pool = head_pool

        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=dropout_p)
        # 分類ヘッドは logit 出力のため BN/ReLU なしの Conv3d 単独 (CAM 重みを兼用)。
        # 元ヘッドの Linear(2048→C) と等価な演算であるため、重みを reshape して移植可能。
        self.head_conv = nn.Conv3d(
            X3D_HEAD_CHANNELS, num_classes, kernel_size=1)
        self._match_bn_momentum()

    def default_checkpoint_blocks(self):
        """GRAD_CHECKPOINT_BLOCKS 未指定時 (checkpoint_sequential) に再計算する前方ブロック数 — デフォルト 4。"""
        n = len(self.features)
        seg = max(1, min(GRAD_CHECKPOINT_SEGMENTS, n))
        return (n // seg) * (seg - 1)

    def _match_bn_momentum(self):
        """チェックポインティング範囲がデフォルトと異なるブロックの BN momentum を調整する (上記の GRAD_CHECKPOINT_BLOCKS 注記参照)。

        同一バッチ統計で momentum m を2回 = 1-(1-m)² を1回。デフォルトで再計算対象だったブロックが除外された場合は
        引き上げ、デフォルトで対象外だったブロックが含まれた場合は 1-√(1-m) に引き下げる。momentum は重みファイルに保存されないため推論には影響しない。
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
        """CAM が特徴マップを取得する層 — ヘッド拡張部の出力 (2048チャネル、空間 7×7 を維持)。

        property として定義しているためモジュールが重複登録されない (重みキーに影響なし)。inference.py が
        `model.cam_layer.register_forward_hook(...)` で特徴マップを取得する。
        head_conv.weight (2048チャネル) とチャネル数が一致することで CAM の重み付き和が成立する。
        """
        return self.head_pool

    def forward(self, x):
        if self.training and GRAD_CHECKPOINT and GRAD_CHECKPOINT_BLOCKS is not None:
            # 前方 GRAD_CHECKPOINT_BLOCKS 個のみブロック単位でチェックポインティング (checkpoint_sequential と等価な計算)
            for i, block in enumerate(self.features):
                x = (checkpoint(block, x, use_reentrant=False)
                     if i < GRAD_CHECKPOINT_BLOCKS else block(x))
        elif self.training and GRAD_CHECKPOINT:
            # バックボーンのみチェックポインティング。head_pool は CAM hook の対象であるため除外。
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
        """forward と同一の logits (B, C) + グリッド別クラス活性化マップ (CAM) (B, C, 7, 7) — 推論専用。

        CAM は分類ヘッド (1×1×1 Conv) を空間平均の「前」に 7×7 グリッドごとに適用したものである。平均と 1×1 Conv は
        順序を入れ替えても同一であるため、グリッド平均がそのまま logits となり、グリッドごとの 'A − S' を確認することで
        そのグリッドがどれだけ事故側に寄与したかを可視化できる。判定のために既に計算されている特徴マップを利用するため、
        追加コストは 49 グリッドの 1×1 乗算のみであり、高速ランナー (ONNX・OpenVINO) でも両出力を同時に生成可能 (infer_runtime.py)。
        logits は forward と同一経路 (平均 → Conv) で計算されるため判定結果に差異は生じない。
        """
        x = self.head_pool(self.features(x))            # (B, 2048, 1, 7, 7)
        cam = self.head_conv(x)[:, :, 0]                # (B, C, 7, 7)
        x = self.head_conv(self.dropout(self.avg_pool(x)))
        return x.squeeze(-1).squeeze(-1).squeeze(-1), cam
