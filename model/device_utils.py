import torch


def get_device(device_type):
    """指定された device_type("cuda"/"cpu") に応じて適切なデバイスを返却します。

    学習時は config.TRAIN_DEVICE_TYPE、推論時は config.INFER_DEVICE_TYPE の値を
    引数として渡します。NVIDIA CUDA と AMD ROCm の双方が "cuda" として設定されます。
    ROCm PyTorch ビルドでは torch.cuda.is_available() が True を返すため、
    個別の条件分岐を行わずに同様に動作します。
    """
    if device_type == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print(
            "\n[警告] DEVICE_TYPE = 'cuda' ですが GPU が検出されないため、CPU で実行します。\n"
            "原因および対処方法:\n"
            "  1. ROCm PyTorch が未インストール\n"
            "     → pip install torch torchvision "
            "--index-url https://download.pytorch.org/whl/rocm6.2\n"
            "  2. CUDA PyTorch が未インストール\n"
            "     → pip install torch torchvision "
            "--index-url https://download.pytorch.org/whl/cu121\n"
            "  3. WSL2 で GPU パススルーが未設定\n"
            "     → /proc/driver/nvidia または /dev/dri デバイスの存在有無を確認\n"
            "  インストール状態の確認: python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\"\n"
        )
        return torch.device("cpu")
    return torch.device("cpu")


def is_rocm() -> bool:
    """現在の PyTorch ビルドが ROCm(HIP) ベースであるか否か。

    ROCm ビルドでは torch.version.hip がバージョン文字列を返し、
    NVIDIA CUDA ビルドでは None を返します。
    """
    return torch.version.hip is not None


def predict_batch_size(device) -> int:
    """本番推論バッチサイズ — GPU は config.PREDICT_INFER_BATCH_SIZE_GPU、その他(CPU)は _CPU。"""
    import config
    size = config.PREDICT_INFER_BATCH_SIZE_GPU if is_cuda_like(device) else config.PREDICT_INFER_BATCH_SIZE_CPU
    return max(1, int(size))


def is_cuda_like(device) -> bool:
    """AMP・cuDNN・pin_memory 等の CUDA 系機能を使用可能なデバイスであるか否か。

    NVIDIA CUDA および ROCm(HIP) PyTorch ビルドの双方でデバイス名が "cuda" となるため
    True を返します。CPU の場合は False を返します。
    """
    return str(device).startswith("cuda")


def is_channels_last_3d_supported(device) -> bool:
    """channels_last_3d メモリフォーマットを安全に使用可能であるか否か。

    ROCm(HIP) は channels_last_3d を完全にはサポートしておらず、
    一部の演算でエラーが発生する可能性があるため無効化します。
    NVIDIA CUDA 環境でのみ有効化します。
    """
    if not is_cuda_like(device):
        return False
    if is_rocm():
        return False
    return True
