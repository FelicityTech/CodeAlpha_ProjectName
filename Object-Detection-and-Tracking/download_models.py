"""
Downloads the YOLOv4-tiny model files (cfg, weights, class names) into
models/ if they aren't already present. Model weights are large binary
files that don't always survive being committed to git or copied between
environments — run this if models/yolov4-tiny.weights is missing.
"""
import urllib.request
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

FILES = {
    "yolov4-tiny.cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
    "coco.names": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names",
    "yolov4-tiny.weights": "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights",
}


def download():
    for filename, url in FILES.items():
        dest = MODEL_DIR / filename
        if dest.exists() and dest.stat().st_size > 0:
            print(f"Already present, skipping: {filename}")
            continue
        print(f"Downloading {filename} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"  -> saved to {dest} ({dest.stat().st_size / 1024:.0f} KB)")

    print("Done.")


if __name__ == "__main__":
    download()
