"""CLI commands for the project."""

import fire


def train():
    """Run model training."""
    # TODO: implement training
    print("Training will be implemented here")


def predict(image_path: str):
    """Run prediction on an image."""
    # TODO: implement prediction
    print(f"Will predict on: {image_path}")


def download_data():
    """Download the dataset."""
    # TODO: implement data download
    print("Will download DocLayNet dataset")


def main():
    """Main function."""
    fire.Fire(
        {
            "train": train,
            "predict": predict,
            "download": download_data,
        }
    )


if __name__ == "__main__":
    main()
