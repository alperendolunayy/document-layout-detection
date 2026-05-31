"""CLI entry point for docuvision."""

import json
import subprocess
from pathlib import Path

import fire
from hydra import compose, initialize_config_dir


def _get_config(overrides=None):
    """Load hydra config."""
    config_dir = str(Path(__file__).parent.parent / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config", overrides=overrides or [])
    return cfg


def train(subset_ratio=None, max_epochs=None, batch_size=None):
    """Run DETR training."""
    # download model checkpoint and data if needed
    _ensure_data()

    overrides = []
    if subset_ratio is not None:
        overrides.append(f"data.subset_ratio={subset_ratio}")
    if max_epochs is not None:
        overrides.append(f"training.max_epochs={max_epochs}")
    if batch_size is not None:
        overrides.append(f"data.batch_size={batch_size}")

    cfg = _get_config(overrides)

    from docuvision.training.train import run_training

    run_training(cfg)


def _ensure_data():
    """Make sure dataset is available before training."""
    data_dir = Path("data/doclaynet")
    if not data_dir.exists() or not any(data_dir.iterdir()):
        print("Dataset not found. Running download...")
        download_data()


def download_data():
    """Download data and model using DVC, with fallback to manual download."""
    # try DVC pull first for model checkpoint
    try:
        result = subprocess.run(
            ["dvc", "pull"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("Model checkpoint downloaded with DVC.")
        else:
            print("DVC pull failed, checkpoint may need manual download.")
    except FileNotFoundError:
        print("DVC not installed. Install with: poetry add dvc")

    # download dataset from public source if not present
    data_dir = Path("data/doclaynet")
    if data_dir.exists() and any(data_dir.iterdir()):
        print("Dataset already exists.")
        return

    print("Downloading DocLayNet dataset (~28GB)...")
    data_dir.mkdir(parents=True, exist_ok=True)
    url = "https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_core.zip"
    try:
        subprocess.run(["wget", "-q", "--show-progress", url, "-O", "doclaynet.zip"], check=True)
    except FileNotFoundError:
        subprocess.run(["curl", "-L", "-o", "doclaynet.zip", url], check=True)
    subprocess.run(["unzip", "-q", "doclaynet.zip", "-d", "data/doclaynet"], check=True)
    Path("doclaynet.zip").unlink(missing_ok=True)
    print("Dataset ready!")


def predict(image_path: str, checkpoint: str = "checkpoints/last.ckpt", threshold: float = 0.5):
    """Run prediction on a document image."""
    image_file = Path(image_path)
    if not image_file.exists():
        print(f"Image not found: {image_file}")
        return

    ckpt_file = Path(checkpoint)
    if not ckpt_file.exists():
        print("Checkpoint not found. Trying dvc pull...")
        try:
            subprocess.run(["dvc", "pull"], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"Could not download checkpoint. Place it at: {ckpt_file}")
            return

    from docuvision.inference import LayoutDetector

    detector = LayoutDetector(checkpoint_path=str(ckpt_file))
    detections = detector.predict(str(image_file), threshold=threshold)
    print(f"Inference time: {detector.last_inference_time_ms}ms")

    print(f"\nFound {len(detections)} elements in {image_file.name}:\n")
    for det in detections:
        print(f"  {det['class_name']:20s} confidence: {det['score']:.3f}  box: {det['box']}")

    # save annotated image
    from PIL import Image

    image = Image.open(image_file).convert("RGB")
    annotated = detector.draw_detections(image, detections)
    output_path = image_file.parent / f"{image_file.stem}_detected{image_file.suffix}"
    annotated.save(output_path)
    print(f"\nAnnotated image saved to: {output_path}")

    # save detections as json
    json_path = image_file.parent / f"{image_file.stem}_detections.json"
    with open(json_path, "w") as file:
        json.dump(detections, file, indent=2)
    print(f"Detections saved to: {json_path}")


def demo():
    """Launch Gradio web demo."""
    cfg = _get_config()

    from docuvision.demo import launch_demo

    launch_demo(
        checkpoint=cfg.serving.checkpoint_path,
        port=cfg.serving.demo_port,
    )


def serve(host: str = None, port: int = None):
    """Start FastAPI server."""
    cfg = _get_config()
    actual_host = host or cfg.serving.host
    actual_port = port or cfg.serving.port

    import uvicorn

    uvicorn.run("docuvision.api:app", host=actual_host, port=actual_port, reload=False)


def main():
    """Main entry point."""
    fire.Fire(
        {
            "train": train,
            "download": download_data,
            "predict": predict,
            "demo": demo,
            "serve": serve,
        }
    )


if __name__ == "__main__":
    main()
