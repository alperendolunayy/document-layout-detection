"""CLI entry point for docuvision."""

import json
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


def download_data():
    """Download dataset using DVC."""
    try:
        import dvc.api

        dvc.api.pull()
        print("Data downloaded with DVC.")
    except ImportError:
        print("DVC not installed. Install with: poetry add dvc")


def predict(image_path: str, checkpoint: str = "checkpoints/last.ckpt", threshold: float = 0.5):
    """Run prediction on a document image.

    Args:
        image_path: path to the document image
        checkpoint: path to model checkpoint
        threshold: confidence threshold for detections
    """
    image_file = Path(image_path)
    if not image_file.exists():
        print(f"Image not found: {image_file}")
        return

    ckpt_file = Path(checkpoint)
    if not ckpt_file.exists():
        print(f"Checkpoint not found: {ckpt_file}")
        return

    from docuvision.inference import LayoutDetector

    detector = LayoutDetector(checkpoint_path=str(ckpt_file))
    detections = detector.predict(str(image_file), threshold=threshold)

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
    with open(json_path, "w") as f:
        json.dump(detections, f, indent=2)
    print(f"Detections saved to: {json_path}")


def demo():
    """Launch Gradio web demo."""
    from docuvision.demo import launch_demo

    launch_demo()


def serve(host: str = "0.0.0.0", port: int = 8000):
    """Start FastAPI server."""
    import uvicorn

    uvicorn.run("docuvision.api:app", host=host, port=port, reload=False)


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
