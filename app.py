from __future__ import annotations

import html
import io
import os
import re
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import pandas as pd

from src.inference import DeepfakeDetector


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

try:
    detector = DeepfakeDetector()
    DETECTOR_ERROR = None
except Exception as exc: 
    detector = None
    DETECTOR_ERROR = str(exc)


GREEN = "#15803d"
DARK = "#172033"
MUTED = "#272d34"
LIGHT_GREEN = "#39d48b"
BORDER = "#e2e8f0"

CSS = f"""
/* Force theme CSS variables across all Gradio internal shadow components */
:root, .dark, body {{
    --body-text-color: #0f172a !important;
    --body-text-color-subdued: #334155 !important;
    --background-fill-primary: #ffffff !important;
    --background-fill-secondary: #f8fafc !important;
    --border-color-primary: #e2e8f0 !important;

    /* Table CSS variables override */
    --table-text-color: #0f172a !important;
    --table-row-focus: #f1f5f9 !important;
    --table-even-background-fill: #ffffff !important;
    --table-odd-background-fill: #f8fafc !important;
}}

/* Force overall container background */
body, .gradio-container {{
    background-color: #f8fafc !important;
    color: #0f172a !important;
    max-width: 1180px !important;
    margin: 0 auto !important;
}}

/* Force Markdown Text & Headings Visibility */
.gradio-container p, 
.gradio-container span, 
.gradio-container h1, 
.gradio-container h2, 
.gradio-container h3, 
.gradio-container h4,
.prose, .prose * {{
    color: #0f172a !important;
}}

/* Hero Section */
.hero {{
    padding: 30px 32px;
    border-radius: 20px;
    background: linear-gradient(120deg, #14532d 0%, #15803d 58%, #0f766e 100%) !important;
    margin-bottom: 18px;
}}
.hero h1 {{ margin: 0 0 8px 0; font-size: clamp(1.7rem, 4vw, 2.35rem); color: #ffffff !important; }}
.hero p {{ margin: 0; color: #f0fdf4 !important; font-size: 1rem; line-height: 1.6; max-width: 780px; }}
.eyebrow {{
    text-transform: uppercase; letter-spacing: .12em; font-size: .75rem;
    font-weight: 800; color: #bbf7d0 !important; margin-bottom: 8px;
}}

/* Cards & Prediction Output */
.result-card {{
    border: 1px solid {BORDER}; border-radius: 18px; padding: 24px;
    background: #ffffff !important; color: {DARK} !important; box-shadow: 0 8px 24px rgba(15,23,42,.04);
}}
.result-card.real {{ border-top: 5px solid {GREEN}; background: #f0fdf4 !important; }}
.result-card.fake {{ border-top: 5px solid #b91c1c; background: #fef2f2 !important; }}
.result-card.unsure {{ border-top: 5px solid #d97706; background: #fffbeb !important; }}
.result-card.error {{ border-top: 5px solid #64748b; background: #f8fafc !important; }}
.result-title {{ font-size: 1.8rem; font-weight: 800; margin: 0 0 6px 0; color: #0f172a !important; }}
.result-sub {{ margin: 0 0 18px 0; line-height: 1.55; color: #334155 !important; }}

/* Metric Progress Bar */
.metric-label {{
    display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap;
    margin: 14px 0 6px; font-size: .9rem; font-weight: 650; color: #334155 !important;
}}
.metric-label span {{ color: #334155 !important; }}
.bar {{ height: 12px; border-radius: 999px; overflow: hidden; background: #e2e8f0; display: flex; }}
.bar > div {{ height: 100%; }}
.small-note {{ margin-top: 14px; color: #475569 !important; font-size: .84rem; line-height: 1.5; }}

/* Stat Summary Cards */
.stats {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 8px 0 20px; }}
.stat {{
    border: 1px solid #a7f3d0; background: #f0fdf4 !important; border-radius: 15px;
    padding: 16px; color: #14532d !important; min-width: 0;
}}
.stat b {{ display: block; font-size: 1.7rem; font-weight: 800; line-height: 1.2; margin-bottom: 5px; color: #14532d !important; }}
.stat span {{ display: block; font-size: .82rem; font-weight: 600; color: #166534 !important; line-height: 1.35; }}

/* Force Dataframe Text Visibility */
table, tr, td, th, .cell-wrap {{
    background-color: #ffffff !important;
    color: #0f172a !important;
}}
th, th * {{
    background-color: #f1f5f9 !important;
    color: #0f172a !important;
    font-weight: 700 !important;
}}

/* Disclaimer Box */
.disclaimer {{
    border: 1px solid #fde68a; background: #fffbeb !important; color: #78350f !important;
    border-radius: 12px; padding: 12px 15px; font-size: .88rem; line-height: 1.5;
}}
.disclaimer * {{ color: #78350f !important; }}

@media (max-width: 760px) {{
    .stats {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .hero {{ padding: 24px 20px; }}
}}
"""


def result_card(title: str, subtitle: str, kind: str = "error", body: str = "") -> str:
    """Return a styled, escaped result card."""
    return (
        f'<section class="result-card {kind}">'
        f'<p class="result-title">{html.escape(title)}</p>'
        f'<p class="result-sub">{html.escape(subtitle)}</p>'
        f"{body}</section>"
    )


def progress_bar(left_pct: float, left_color: str, right_color: str) -> str:
    left_pct = max(0.0, min(100.0, float(left_pct)))
    right_pct = 100.0 - left_pct
    return (
        '<div class="bar" role="img" aria-label="Proportion bar">'
        f'<div style="width:{left_pct:.1f}%;background:{left_color}"></div>'
        f'<div style="width:{right_pct:.1f}%;background:{right_color}"></div>'
        "</div>"
    )


IDLE = result_card(
    "Ready to analyze",
    "Upload a face video, then select Analyze video to see the model's prediction.",
    "error",
    '<p class="small-note">The result is a model estimate, not proof of authenticity.</p>',
)


def analyze_video(video_path: str | None):
    if not video_path:
        return IDLE, []

    if detector is None:
        message = DETECTOR_ERROR or "The detector could not be initialized."
        return result_card("Model unavailable", message, "error"), []

    try:
        result: dict[str, Any] = detector.predict_video(video_path)
    except Exception as exc:  # noqa: BLE001 - report inference errors to the user
        return result_card(
            "Analysis failed",
            f"{type(exc).__name__}: {exc}",
            "error",
            '<p class="small-note">Check the model files, video format, and server logs.</p>',
        ), []

    if not isinstance(result, dict):
        return result_card("Unexpected model response", "The detector returned an unsupported result.", "error"), []

    if result.get("status") == "error":
        message = str(result.get("message", "The video could not be analyzed."))
        return result_card("Could not analyze video", message, "error"), []

    try:
        fake_probability = float(result["fake_probability"])
        fake_probability = max(0.0, min(1.0, fake_probability))
    except (KeyError, TypeError, ValueError):
        return result_card(
            "Invalid model output",
            "The detector did not return a valid fake probability.",
            "error",
        ), []

    alpha = result.get("gate_alpha")
    try:
        alpha = max(0.0, min(1.0, float(alpha)))
    except (TypeError, ValueError):
        alpha = None

    if 0.35 < fake_probability < 0.65:
        title, kind = "Uncertain prediction", "unsure"
        subtitle = (
            f"The model estimates a {fake_probability:.1%} probability of manipulation. "
            "The score is near the decision boundary, so treat this result cautiously."
        )
    elif fake_probability >= 0.65:
        title, kind = "Likely manipulated", "fake"
        subtitle = f"The model estimates a {fake_probability:.1%} probability that this video is fake."
    else:
        title, kind = "Likely authentic", "real"
        subtitle = f"The model estimates a {(1 - fake_probability):.1%} probability that this video is real."

    real_pct = (1.0 - fake_probability) * 100.0
    fake_pct = fake_probability * 100.0
    body = (
        '<div class="metric-label"><span>Estimated class probabilities</span>'
        f'<span>Real {real_pct:.1f}% · Fake {fake_pct:.1f}%</span></div>'
        + progress_bar(real_pct, GREEN, "#b91c1c")
    )

    if alpha is not None:
        body += (
            '<div class="metric-label"><span>Fusion gate weighting</span>'
            f'<span>Spatial {alpha:.0%} · Frequency {(1 - alpha):.0%}</span></div>'
            + progress_bar(alpha * 100, "#0f766e", "#64748b")
            + '<p class="small-note">The gate value describes the model’s spatial/frequency feature weighting; '
              "it is not a confidence score.</p>"
        )

    face_count = result.get("faces_detected")
    if face_count is not None:
        body += (
            f'<p class="small-note">Faces detected in {html.escape(str(face_count))} of 8 sampled frames. '
            "Performance may vary with video quality, lighting, compression, and face visibility.</p>"
        )

    faces = result.get("faces")
    if faces is None:
        faces = []
    elif isinstance(faces, np.ndarray):
        faces = list(faces) if faces.ndim == 4 else []
    else:
        faces = list(faces)

    return result_card(title, subtitle, kind, body), list(faces)


FALLBACK_CSV = """model,AUC,acc,REAL_recall,FAKE_recall,bal_acc
spatial,0.9882 [0.979-0.994],0.9617 [0.947-0.973],0.9257 [0.879-0.964],0.9689 [0.956-0.981],0.9473 [0.923-0.967]
frequency,0.9692 [0.957-0.980],0.9257 [0.909-0.941],0.7770 [0.712-0.840],0.9554 [0.941-0.970],0.8662 [0.832-0.898]
late_fusion,0.9904 [0.983-0.996],0.9673 [0.955-0.979],0.9257 [0.876-0.965],0.9757 [0.964-0.985],0.9507 [0.925-0.971]
gated,0.9889 [0.981-0.994],0.9595 [0.947-0.972],0.9122 [0.864-0.956],0.9689 [0.957-0.981],0.9405 [0.916-0.963]
"""

MODEL_NAMES = {
    "spatial": "Spatial only",
    "frequency": "Frequency only",
    "late_fusion": "Late fusion",
    "gated": "Gated fusion",
}
METRIC_COLUMNS = [
    ("AUC", "ROC-AUC"),
    ("acc", "Accuracy"),
    ("REAL_recall", "REAL recall"),
    ("FAKE_recall", "FAKE recall"),
    ("bal_acc", "Balanced accuracy"),
]


def load_metrics() -> pd.DataFrame:
    path = RESULTS / "final_metrics_table.csv"
    try:
        if path.exists():
            frame = pd.read_csv(path, dtype=str)
        else:
            frame = pd.read_csv(io.StringIO(FALLBACK_CSV), dtype=str)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        frame = pd.read_csv(io.StringIO(FALLBACK_CSV), dtype=str)

    required = {"model", *(key for key, _ in METRIC_COLUMNS)}
    if not required.issubset(frame.columns):
        frame = pd.read_csv(io.StringIO(FALLBACK_CSV), dtype=str)
    return frame


def parse_metric(value: Any) -> tuple[float | None, tuple[float, float] | None]:
    if value is None or pd.isna(value) or not str(value).strip():
        return None, None
    match = re.fullmatch(
        r"\s*([0-9]*\.?[0-9]+)\s*(?:\[?\s*([0-9]*\.?[0-9]+)\s*[-–]\s*([0-9]*\.?[0-9]+)\s*\]?)?\s*",
        str(value),
    )
    if not match:
        raise ValueError(f"Invalid metric value: {value!r}")
    number = float(match.group(1))
    interval = (float(match.group(2)), float(match.group(3))) if match.group(2) else None
    return number, interval


def metrics_table() -> pd.DataFrame:
    frame = load_metrics()
    rows: list[dict[str, str]] = []
    for _, record in frame.iterrows():
        row = {"Model": MODEL_NAMES.get(str(record.get("model", "")), str(record.get("model", "")))}
        for key, label in METRIC_COLUMNS:
            value, interval = parse_metric(record.get(key))
            if value is None:
                row[label] = "Not reported"
            elif key == "AUC":
                row[label] = f"{value:.4f}"
                if interval:
                    row[label] += f" ({interval[0]:.3f}–{interval[1]:.3f})"
            else:
                row[label] = f"{value:.2%}"
                if interval:
                    row[label] += f" ({interval[0]:.1%}–{interval[1]:.1%})"
        rows.append(row)
    return pd.DataFrame(rows)


def summary_cards() -> str:
    frame = load_metrics()
    selected = frame[frame["model"].astype(str).str.lower() == "gated"]
    if selected.empty:
        return '<p class="small-note">Gated-fusion metrics are not available.</p>'

    record = selected.iloc[0]
    cards = []
    for key, label, formatter in [
        ("AUC", "ROC-AUC", lambda x: f"{x:.4f}"),
        ("acc", "Test accuracy", lambda x: f"{x:.2%}"),
        ("REAL_recall", "REAL recall", lambda x: f"{x:.2%}"),
        ("FAKE_recall", "FAKE recall", lambda x: f"{x:.2%}"),
    ]:
        value, _ = parse_metric(record.get(key))
        display = formatter(value) if value is not None else "N/A"
        cards.append(
            f'<div class="stat"><b>{html.escape(display)}</b>'
            f'<span>{html.escape(label)}</span></div>'
        )
    return '<div class="stats">' + "".join(cards) + "</div>"


def chart_paths() -> list[str]:
    candidates = [
        RESULTS / "figs" / "roc_v2.png",
        RESULTS / "figs" / "per_method_v2.png",
        RESULTS / "figs" / "confusion_gated_v2.png",
        RESULTS / "figs" / "gate_alpha_v2.png",
    ]
    return [str(path) for path in candidates if path.is_file()]


DISABLE_DARK_MODE_JS = """
() => {
    document.documentElement.classList.remove('dark');
    document.body.classList.remove('dark');
    const observer = new MutationObserver(() => {
        document.documentElement.classList.remove('dark');
        document.body.classList.remove('dark');
    });
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
}
"""

THEME = gr.themes.Soft(
    primary_hue="green",
    secondary_hue="teal",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)
try:
    GRADIO_MAJOR = int(gr.__version__.split(".")[0])
except Exception:  # noqa: BLE001
    GRADIO_MAJOR = 5
STYLE_KWARGS = {"theme": THEME, "css": CSS, "js": DISABLE_DARK_MODE_JS}  # js forces light mode

with gr.Blocks(title="Multimedia Authentication", **({} if GRADIO_MAJOR >= 6 else STYLE_KWARGS)) as demo:
    gr.HTML(
        """
        <header class="hero">
          <div class="eyebrow">AI research demo · FaceForensics++ C23</div>
          <h1>Multimedia Authentication</h1>
          <p>Inspect a face video with two complementary signals: visual appearance and
          high-frequency patterns. The model returns an estimated real/fake probability
          and, when available, the spatial/frequency gate weighting.</p>
        </header>
        """
    )

    with gr.Tabs():
        with gr.Tab("Analyze video"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=5, min_width=280):
                    video = gr.Video(label="Upload a video", height=300)
                    analyze_button = gr.Button("Analyze video", variant="primary", size="lg")
                    gr.Markdown(
                        "For best results, use a short clip with a clearly visible face. "
                        "The app samples up to eight frames."
                    )
                with gr.Column(scale=6, min_width=300):
                    verdict = gr.HTML(IDLE)
            gallery = gr.Gallery(
                label="Sampled face crops",
                columns=4,
                height=220,
                object_fit="contain",
                show_label=True,
            )
            analyze_button.click(analyze_video, inputs=video, outputs=[verdict, gallery])

        with gr.Tab("Model performance"):
            gr.Markdown(
                "### Held-out test results\n"
                "Metrics below are from the project's reported evaluation. The gated-fusion model is the "
                "model used for inference in this demo. Values in brackets are 95% bootstrap confidence intervals "
                "over the 888 test videos."
            )
            gr.HTML(summary_cards())
            gr.Markdown("#### Model comparison")
            gr.Dataframe(value=metrics_table(), interactive=False, wrap=True)
            charts = chart_paths()
            if charts:
                gr.Gallery(
                    value=charts,
                    label="Evaluation charts",
                    columns=2,
                    height=480,
                    object_fit="contain",
                )
            gr.HTML(
                '<div class="disclaimer">Test-set performance does not guarantee performance on every video '
                "or on videos from a different source. Avoid interpreting probabilities as calibrated certainty "
                "unless calibration has been evaluated separately.</div>"
            )


if __name__ == "__main__":
    demo.launch(**(STYLE_KWARGS if GRADIO_MAJOR >= 6 else {}))
