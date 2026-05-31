"""Gradio web demo for document layout detection."""

from pathlib import Path

import gradio as gr

from docuvision.inference import LayoutDetector

DEFAULT_CHECKPOINT = "checkpoints/last.ckpt"


def create_demo(detector):
    """Create the Gradio interface.

    Args:
        detector: LayoutDetector instance
    """

    def detect(image, threshold):
        """Run detection and return annotated image with results."""
        if image is None:
            return None, "Please upload an image."

        detections = detector.predict(image, threshold=threshold)
        annotated = detector.draw_detections(image.copy(), detections)

        # build results text
        lines = [f"Found {len(detections)} elements:\n"]
        for det in detections:
            lines.append(f"  {det['class_name']:20s}  confidence: {det['score']:.3f}")

        return annotated, "\n".join(lines)

    demo = gr.Interface(
        fn=detect,
        inputs=[
            gr.Image(type="pil", label="Upload Document Image"),
            gr.Slider(
                minimum=0.1,
                maximum=0.9,
                value=0.5,
                step=0.05,
                label="Confidence Threshold",
            ),
        ],
        outputs=[
            gr.Image(type="pil", label="Detected Layout"),
            gr.Textbox(label="Detection Results", lines=10),
        ],
        title="DocuVision - Document Layout Detection",
        description="Upload a document image to detect layout elements like tables, "
        "figures, text blocks, headers, and more. Powered by DETR fine-tuned on DocLayNet.",
        examples=[],
        flagging_mode="never",
    )

    return demo


def launch_demo(checkpoint=None, share=False, port=8080):
    """Launch the Gradio demo.

    Args:
        checkpoint: path to model checkpoint
        share: whether to create a public link
        port: port number for the demo server
    """
    ckpt = checkpoint or DEFAULT_CHECKPOINT
    if not Path(ckpt).exists():
        print(f"Checkpoint not found: {ckpt}")
        print("Please provide a valid checkpoint path.")
        return

    print(f"Loading model from {ckpt}...")
    detector = LayoutDetector(checkpoint_path=ckpt)

    demo = create_demo(detector)
    demo.launch(share=share, server_name="127.0.0.1", server_port=port)


if __name__ == "__main__":
    launch_demo()
