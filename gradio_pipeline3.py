"""
=============================================================
  PIPELINE 3 — GRADIO FRONTEND
  A polished UI for Cloud SD3 512×512 → RealESRGAN 2048×2048
=============================================================
=============================================================
"""

import base64
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests

import gradio as gr

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_CLOUD_URL = "http://10.44.67.101:8765"
API_KEY            = os.environ.get("CLOUD_API_KEY", "pipeline3-cloud-key")
MODEL_PATH         = os.path.join(os.path.dirname(__file__), "models", "RealESRGAN_x4plus.pth")
ESRGAN_SCALE       = 4
TILE_SIZE          = 256
MAX_WAIT_SEC       = 600
DEFAULT_POLL_SEC   = 3
SAVE_DIR           = Path("./output")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("GradioUI")

# ── Shared state ──────────────────────────────────────────────────────────────
_upsampler   = None   # lazy-loaded once
_session     = None   # shared requests.Session


def get_session():
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers["X-API-Key"] = API_KEY
    return _session


# ── RealESRGAN loader ─────────────────────────────────────────────────────────
def load_realesrgan():
    global _upsampler
    if _upsampler is not None:
        return _upsampler, None

    try:
        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
    except Exception as e:
        return None, f"RealESRGAN import failed: {e}\n\npip install torch basicsr realesrgan"

    if not os.path.exists(MODEL_PATH):
        return None, (
            f"Model weights not found at: {MODEL_PATH}\n\n"
            "Download with:\n"
            "  mkdir -p models\n"
            "  wget -O models/RealESRGAN_x4plus.pth \\\n"
            "    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                     num_block=23, num_grow_ch=32, scale=ESRGAN_SCALE)
    _upsampler = RealESRGANer(
        scale=ESRGAN_SCALE,
        model_path=MODEL_PATH,
        model=model,
        tile=TILE_SIZE,
        tile_pad=10,
        pre_pad=0,
        half=torch.cuda.is_available(),
        device=device,
    )
    return _upsampler, None


# ── Cloud helpers ─────────────────────────────────────────────────────────────
def cloud_health(cloud_url):
    try:
        r = get_session().get(f"{cloud_url}/health", timeout=5)
        d = r.json()
        return d.get("status") == "healthy", d
    except Exception as e:
        return False, {"error": str(e)}


def cloud_queue_info(cloud_url):
    try:
        d = get_session().get(f"{cloud_url}/queue/info", timeout=5).json()
        q, j, s = d["queue"], d["jobs"], d["stats"]
        lines = [
            f"**Node:** {d.get('cloud_node', 'N/A')}  |  **Workers:** {d.get('workers')}",
            f"**Device:** {d.get('device', '?')}  |  **GPU busy:** {d.get('gpu_busy', '?')}",
            f"**Queue:** {q['depth']}/{q['max_depth']} ({q['fill_pct']}% full)",
            f"**Jobs —** queued: {j['queued']}  processing: {j['processing']}  "
            f"done: {j['done']}  failed: {j['failed']}",
            f"**Avg gen time:** {s.get('avg_latency_sec', 'N/A')} s",
        ]
        return "\n\n".join(lines)
    except Exception as e:
        return f"Could not fetch queue info: {e}"


def submit_job(cloud_url, prompt, negative, steps, guidance, client_id):
    r = get_session().post(
        f"{cloud_url}/generate",
        json={
            "prompt": prompt,
            "negative_prompt": negative,
            "steps": steps,
            "guidance_scale": guidance,
            "client_id": client_id,
        },
        timeout=10,
    )
    if r.status_code == 429: raise RuntimeError("Rate limited by cloud node")
    if r.status_code == 503: raise RuntimeError("Cloud queue is full — try again later")
    if r.status_code != 202: raise RuntimeError(f"Submit failed [{r.status_code}]: {r.text}")
    return r.json()["job_id"]


def poll_job(cloud_url, job_id):
    t0 = time.time()
    while True:
        if time.time() - t0 > MAX_WAIT_SEC:
            raise TimeoutError(f"Timed out after {MAX_WAIT_SEC}s")
        job = get_session().get(f"{cloud_url}/status/{job_id}", timeout=10).json()
        st  = job["status"]
        if st == "DONE":    return job
        if st == "FAILED":  raise RuntimeError(f"Cloud job failed: {job.get('error')}")
        if st == "CANCELLED": raise RuntimeError("Job was cancelled")
        time.sleep(DEFAULT_POLL_SEC)


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(cloud_bgr, upscaled_bgr):
    try:
        from skimage.metrics import peak_signal_noise_ratio as psnr
        from skimage.metrics import structural_similarity as ssim
        h, w = upscaled_bgr.shape[:2]
        base  = cv2.resize(cloud_bgr, (w, h), interpolation=cv2.INTER_CUBIC)
        b_rgb = cv2.cvtColor(base,         cv2.COLOR_BGR2RGB)
        u_rgb = cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2RGB)
        pv    = psnr(b_rgb, u_rgb, data_range=255)
        sv    = ssim(b_rgb, u_rgb, channel_axis=2, data_range=255)
        return pv, sv
    except ImportError:
        return None, None


# ── Main generate function ────────────────────────────────────────────────────
def generate(
    prompt, negative, steps, guidance,
    cloud_url, client_id,
    progress=gr.Progress(track_tqdm=False),
):
    if not prompt.strip():
        raise gr.Error("Please enter a prompt.")

    logs = []

    def log_step(msg):
        logs.append(msg)
        return "\n".join(logs)

    # 1. Load upscaler
    progress(0.05, desc="Loading RealESRGAN weights…")
    upsampler, err = load_realesrgan()
    if err:
        raise gr.Error(err)
    yield None, None, log_step(" RealESRGAN weights loaded"), "", gr.update(interactive=True)

    # 2. Health check
    progress(0.10, desc="Checking cloud node…")
    healthy, hinfo = cloud_health(cloud_url)
    if not healthy:
        raise gr.Error(f"Cloud node not healthy: {hinfo}")
    yield None, None, log_step(f" Cloud healthy — {hinfo.get('cloud_node', cloud_url)}"), "", gr.update(interactive=True)

    # 3. Submit job
    progress(0.15, desc="Submitting job…")
    try:
        job_id = submit_job(cloud_url, prompt, negative, int(steps), guidance, client_id)
    except Exception as e:
        raise gr.Error(str(e))
    yield None, None, log_step(f" Job submitted: {job_id}"), "", gr.update(interactive=True)

    # 4. Poll
    progress(0.20, desc="Waiting for cloud generation…")
    try:
        job = poll_job(cloud_url, job_id)
    except Exception as e:
        raise gr.Error(str(e))

    info     = job["image_info"]
    lat      = job["latency_sec"]
    pay_kb   = info["payload_kb"]
    hint_sz  = info["upscale_hint"]["output_size"]
    yield None, None, log_step(
        f"  Cloud done — {lat}s | {pay_kb} KB | hint → {hint_sz}"
    ), "", gr.update(interactive=True)

    # 5. Decode & save raw 512 image
    progress(0.55, desc="Receiving 512×512 image…")
    jpeg_bytes = base64.b64decode(job["result_b64"])
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = prompt[:20].replace(" ", "_").replace("/", "-")
    raw_path = SAVE_DIR / f"cloud_raw_512_{ts}_{safe}.jpg"
    raw_path.write_bytes(jpeg_bytes)
    cloud_bgr = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
    raw_rgb   = cv2.cvtColor(cloud_bgr, cv2.COLOR_BGR2RGB)

    yield raw_rgb, None, log_step(f"  Raw 512×512 saved → {raw_path.name}"), "", gr.update(interactive=True)

    # 6. Upscale
    progress(0.65, desc="Running RealESRGAN upscale…")
    img_bgr = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
    t0 = time.time()
    upscaled_bgr, _ = upsampler.enhance(img_bgr, outscale=ESRGAN_SCALE)
    up_t = round(time.time() - t0, 2)

    # 7. Save final PNG
    out_path = SAVE_DIR / f"final_2048_{ts}_{safe}.png"
    cv2.imwrite(str(out_path), upscaled_bgr)
    upscaled_rgb = cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2RGB)
    yield raw_rgb, upscaled_rgb, log_step(f"⬆  Upscale done in {up_t}s → {out_path.name}"), "", gr.update(interactive=True)

    # 8. Metrics
    progress(0.95, desc="Computing quality metrics…")
    pv, sv = compute_metrics(cloud_bgr, upscaled_bgr)
    metrics_md = ""
    if pv is not None:
        metrics_md = (
            f"| Metric | Value |\n|---|---|\n"
            f"| PSNR (vs bicubic) | **{pv:.2f} dB** |\n"
            f"| SSIM (vs bicubic) | **{sv:.4f}** |"
        )
        yield raw_rgb, upscaled_rgb, log_step(
            f" PSNR={pv:.2f} dB  SSIM={sv:.4f}"
        ), metrics_md, gr.update(interactive=True)

    # 9. Summary
    progress(1.0, desc="Done!")
    summary = (
        f" **Pipeline 3 complete**\n\n"
        f"- **Prompt:** {prompt[:70]}\n"
        f"- **Cloud gen:** {lat}s  (SD3 Medium 512×512)\n"
        f"- **Payload:** {pay_kb} KB  (JPEG q=85)\n"
        f"- **Upscale:** {up_t}s  (RealESRGAN x4plus)\n"
        f"- **Output:** 2048×2048 PNG  →  `{out_path.name}`"
    )
    yield raw_rgb, upscaled_rgb, summary, metrics_md, gr.update(interactive=True)


# ── Queue info helper ─────────────────────────────────────────────────────────
def check_queue(cloud_url):
    healthy, hinfo = cloud_health(cloud_url)
    status_tag = " Healthy" if healthy else f" Unreachable — {hinfo.get('error', '')}"
    return f"**Status:** {status_tag}\n\n" + cloud_queue_info(cloud_url)


# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:       #0d0d0f;
    --surface:  #141418;
    --border:   #2a2a32;
    --accent:   #e8c97e;
    --accent2:  #7eb8e8;
    --text:     #e8e6e0;
    --muted:    #6b6b78;
    --radius:   10px;
}

body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text) !important;
}

/* Header */
#header {
    text-align: center;
    padding: 2rem 0 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
#header h1 {
    font-family: 'DM Serif Display', serif !important;
    font-size: 2.4rem;
    letter-spacing: -0.02em;
    color: var(--accent);
    margin: 0;
}
#header p {
    color: var(--muted);
    font-size: 0.88rem;
    margin: 0.3rem 0 0;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.05em;
}

/* Panels */
.panel {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}

/* Labels */
label span, .label-wrap span {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: var(--muted) !important;
}

/* Inputs */
textarea, input[type=text], input[type=number] {
    background: #1a1a20 !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 6px !important;
    font-family: 'DM Sans', sans-serif !important;
}
textarea:focus, input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(232,201,126,0.12) !important;
}

/* Sliders */
input[type=range] { accent-color: var(--accent) !important; }

/* Generate button */
#gen-btn {
    background: var(--accent) !important;
    color: #0d0d0f !important;
    font-family: 'DM Serif Display', serif !important;
    font-size: 1.05rem !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.75rem 2rem !important;
    cursor: pointer;
    transition: opacity 0.15s;
    width: 100%;
}
#gen-btn:hover  { opacity: 0.88 !important; }
#gen-btn:active { opacity: 0.75 !important; }

/* Queue button */
#queue-btn {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--accent2) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
    border-radius: 6px !important;
    width: 100%;
}
#queue-btn:hover { border-color: var(--accent2) !important; }

/* Image outputs */
.image-container img {
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
}

/* Log / markdown */
#log-box .prose, #metrics-box .prose {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
    color: var(--text) !important;
    background: #0d0d0f !important;
    padding: 0.8rem !important;
    border-radius: 6px !important;
    min-height: 5rem;
}

/* Queue info */
#queue-info .prose {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
}

/* Accordion */
.accordion { border: 1px solid var(--border) !important; border-radius: 8px !important; }
.accordion-header { background: var(--surface) !important; }

/* Tab bar */
.tab-nav button {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    color: var(--muted) !important;
}
.tab-nav button.selected { color: var(--accent) !important; border-bottom-color: var(--accent) !important; }
"""

# ── UI layout ─────────────────────────────────────────────────────────────────
with gr.Blocks(css=CSS, title="Client-Edge Image Generator") as demo:

    gr.HTML("""
    <div id="header">
        <h1>Cloud-Edge Rendering</h1>
        <p>Cloud SD3 512×512 &nbsp;→&nbsp; RealESRGAN ×4 &nbsp;→&nbsp; 2048×2048</p>
    </div>
    """)

    with gr.Row():
        # ── Left column: controls ──────────────────────────────────────────
        with gr.Column(scale=1, min_width=320, elem_classes="panel"):
            prompt_box = gr.Textbox(
                label="Prompt",
                placeholder="eG: a futuristic city at sunset, cinematic lighting…",
                lines=4,
            )
            negative_box = gr.Textbox(
                label="Negative Prompt",
                value="blurry, low quality, distorted, noisy, artifacts",
                lines=2,
            )

            with gr.Accordion("Advanced settings", open=False):
                steps_slider = gr.Slider(
                    label="Inference Steps",
                    minimum=10, maximum=50, step=1, value=28,
                )
                guidance_slider = gr.Slider(
                    label="Guidance Scale",
                    minimum=1.0, maximum=15.0, step=0.5, value=7.0,
                )
                cloud_url_box = gr.Textbox(
                    label="Cloud URL",
                    value=DEFAULT_CLOUD_URL,
                )
                client_id_box = gr.Textbox(
                    label="Client ID",
                    value="edge-client-01",
                )

            gen_btn = gr.Button("✦ Generate", elem_id="gen-btn", variant="primary")

            gr.Markdown("---")
            queue_btn  = gr.Button("Check Cloud Queue", elem_id="queue-btn")
            queue_info = gr.Markdown("", elem_id="queue-info")

        # ── Right column: output ───────────────────────────────────────────
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("Final  2048×2048"):
                    out_2048 = gr.Image(
                        label="Upscaled Output",
                        show_label=False,
                        height=512,
                        elem_classes="panel",
                    )
                with gr.Tab("Raw  512×512"):
                    out_512 = gr.Image(
                        label="Cloud Raw",
                        show_label=False,
                        height=512,
                        elem_classes="panel",
                    )

            log_md     = gr.Markdown("", elem_id="log-box")
            metrics_md = gr.Markdown("", elem_id="metrics-box")

    # ── Events ────────────────────────────────────────────────────────────────
    gen_btn.click(
        fn=generate,
        inputs=[
            prompt_box, negative_box,
            steps_slider, guidance_slider,
            cloud_url_box, client_id_box,
        ],
        outputs=[out_512, out_2048, log_md, metrics_md, gen_btn],
        show_progress="full",
    )

    queue_btn.click(
        fn=check_queue,
        inputs=[cloud_url_box],
        outputs=[queue_info],
    )


if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)