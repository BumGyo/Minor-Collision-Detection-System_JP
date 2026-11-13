import torch


def get_device(device_type):
    """指定された device_type ("cuda"/"cpu") に応じて適切なデバイスを返します。

    学習時は config.TRAIN_DEVICE_TYPE、推論時は config.INFER_DEVICE_TYPE を引数として渡します。
    NVIDIA CUDA および AMD ROCm の双方が "cuda" に対応します。
    """
    if device_type == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print(
            "\n[警告] DEVICE_TYPE = 'cuda' が指定されましたが、利用可能なGPUが検出されないためCPUで実行します。\n"
            "原因と対処法:\n"
            "  1. ROCm PyTorch 未インストール\n"
            "     → pip install torch torchvision "
            "--index-url https://download.pytorch.org/whl/rocm6.2\n"
            "  2. CUDA PyTorch 未インストール\n"
            "     → pip install torch torchvision "
            "--index-url https://download.pytorch.org/whl/cu121\n"
            "  3. WSL2 GPUパススルー未設定\n"
            "     → /proc/driver/nvidia または /dev/dri の存在を確認\n"
            "  確認コマンド: python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\"\n"
        )
        return torch.device("cpu")
    return torch.device("cpu")


def is_rocm() -> bool:
    """現在の PyTorch ビルドが ROCm(HIP) ベースか判定。"""
    return torch.version.hip is not None


def predict_batch_size(device) -> int:
    """推論バッチサイズ — GPU時は config.PREDICT_INFER_BATCH_SIZE_GPU、CPU時は _CPU。"""
    import config
    size = config.PREDICT_INFER_BATCH_SIZE_GPU if is_cuda_like(device) else config.PREDICT_INFER_BATCH_SIZE_CPU
    return max(1, int(size))


def is_cuda_like(device) -> bool:
    """AMP・cuDNN・pin_memory などの CUDA 互換機能を使用可能なデバイスか判定。"""
    return str(device).startswith("cuda")


def is_channels_last_3d_supported(device) -> bool:
    """channels_last_3d メモリフォーマットが安全に使用可能か判定。

    NVIDIA CUDA 環境でのみ有効化。
    """
    if not is_cuda_like(device):
        return False
    if is_rocm():
        return False
    return True
