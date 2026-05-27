"""CLI entry point for docuvision."""

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


def predict(image_path: str):
    """Run prediction on a document image."""
    image_file = Path(image_path)
    if not image_file.exists():
        print(f"Image not found: {image_file}")
        return

    print(f"Running prediction on: {image_file}")
    # TODO: load model and run inference
    print("Not implemented yet.")


def main():
    """Main entry point."""
    fire.Fire(
        {
            "train": train,
            "download": download_data,
            "predict": predict,
        }
    )


if __name__ == "__main__":
    main()
