"""从 Kaggle 下载 Face Mask Detection 数据集并解压到 data/raw/。"""

import subprocess
import sys
import zipfile
from pathlib import Path


def install_kagglehub() -> None:
    """静默安装 kagglehub 包。"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub", "-q"])


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    dest = project_root / "data" / "raw"

    try:
        import kagglehub  # noqa: F401
    except ImportError:
        print("Installing kagglehub...")
        install_kagglehub()
        import kagglehub

    print(f"Downloading andrewmvd/face-mask-detection to {dest} ...")
    archive_path = kagglehub.dataset_download(
        "andrewmvd/face-mask-detection", path=dest, force_download=True
    )
    archive_path = Path(archive_path)

    # 数据集包含一个 zip 文件，需要解压
    zip_files = list(archive_path.rglob("*.zip"))
    if zip_files:
        print(f"Extracting {len(zip_files)} archive(s)...")
        for zf in zip_files:
            with zipfile.ZipFile(zf, "r") as z:
                z.extractall(dest)
            zf.unlink()
        print("Extraction complete.")

    # 清理 kagglehub 遗留的空目录
    for subdir in sorted(dest.rglob("*"), reverse=True):
        if subdir.is_dir() and not any(subdir.iterdir()):
            subdir.rmdir()

    print("Done. Files:")
    for item in sorted(dest.iterdir()):
        print(f"  {item.name}")


if __name__ == "__main__":
    main()
