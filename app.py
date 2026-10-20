"""Gradio demo.  Run:  python app.py"""

import gradio as gr

from src.inference import DeepfakeDetector

detector = DeepfakeDetector()

DISCLAIMER = (
    "Research prototype trained on a subset of FaceForensics++ (C23). "
    "It is **not** a reliable real-world deepfake detector: on our test set it "
    "correctly recognised only about half of the REAL videos."
)


def analyze_video(video_path):
    if video_path is None:
        return "Please upload a video.", None, ""

    result = detector.predict_video(video_path)
    if result["status"] == "error":
        return result["message"], None, ""

    p_fake = result["fake_probability"]
    headline = (
        f"{result['prediction']}  (score {result['confidence'] * 100:.1f}%)"
    )
    details = (
        f"Frames with a detected face: {result['faces_detected']}\n"
        f"Spatial-branch weight (gate alpha): {result['gate_alpha']:.4f}\n"
        f"Frequency-branch weight: {1 - result['gate_alpha']:.4f}"
    )
    return headline, {"FAKE": p_fake, "REAL": 1.0 - p_fake}, details


with gr.Blocks(title="Multimedia Authentication") as demo:
    gr.Markdown(
        "# Multimedia Authentication\n"
        "### Deepfake video detection with spatial + frequency (DWT) features\n"
        "Upload a short video that shows a clear face."
    )
    video = gr.Video(label="Upload video")
    button = gr.Button("Analyze video", variant="primary")
    verdict = gr.Textbox(label="Result")
    probs = gr.Label(label="Model scores", num_top_classes=2)
    details = gr.Textbox(label="Details", lines=3)
    button.click(analyze_video, inputs=video, outputs=[verdict, probs, details])
    gr.Markdown(DISCLAIMER)

if __name__ == "__main__":
    demo.launch()
