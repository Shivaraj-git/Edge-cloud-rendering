"""
=============================================================
  PIPELINE 3 — EDGE CLIENT
  Cloud-generated 512×512 → RealESRGAN upscale → 2048×2048
=============================================================
  Pipeline:
    1. Health-check cloud node
    2. Submit prompt to cloud  →  cloud generates 512×512 via SD3
    3. Poll until DONE
    4. Receive 512×512 JPEG (base64) from cloud
    5. Decode + save as cloud_raw_512.jpg
    6. RealESRGAN x4plus upscale → 2048×2048 PNG
    7. Save final_2048.png + print quality metrics

  Install:
    pip install requests pillow torch opencv-python basicsr realesrgan scikit-image

  Download model weights (once):
    mkdir -p models
    wget -O models/RealESRGAN_x4plus.pth \
      https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

  Run:
    export CLOUD_API_KEY=pipeline3-cloud-key
    python client_pipeline3.py --prompt "a futuristic city at sunset"
=============================================================
"""

import argparse
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
import torch

DEFAULT_CLOUD_URL = "http://10.44.67.101:8765"
API_KEY = os.environ.get("CLOUD_API_KEY", "pipeline3-cloud-key")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "RealESRGAN_x4plus.pth")
ESRGAN_SCALE = 4  # 512 × 4 = 2048
TILE_SIZE = 256  # set 0 to disable tiling (needs more VRAM)
MAX_WAIT_SEC = 600
DEFAULT_POLL_SEC = 3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("EdgeClient")


# ─────────────────────────────────────────────
#  RealESRGAN LOADER
# ─────────────────────────────────────────────
def load_realesrgan():
    """Load RealESRGAN x4plus. Fails fast with a helpful message if missing."""
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise RuntimeError(f"RealESRGAN import FAILED: {e}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model weights not found at: {MODEL_PATH}\n"
            "  Download:\n"
            "    mkdir -p models\n"
            "    wget -O models/RealESRGAN_x4plus.pth \\\n"
            "      https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"[Upscaler] Device: {device}")

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=ESRGAN_SCALE,
    )
    upsampler = RealESRGANer(
        scale=ESRGAN_SCALE,
        model_path=MODEL_PATH,
        model=model,
        tile=TILE_SIZE,
        tile_pad=10,
        pre_pad=0,
        half=torch.cuda.is_available(),  # fp16 on GPU, fp32 on CPU
        device=device,
    )
    log.info(f"[Upscaler] RealESRGAN x{ESRGAN_SCALE} ready")
    return upsampler


# ─────────────────────────────────────────────
#  UPSCALE
# ─────────────────────────────────────────────
def upscale(upsampler, input_path: Path) -> tuple:
    """
    Upscale the cloud JPEG from 512×512 to 2048×2048.
    Returns (output_bgr_array, elapsed_seconds).
    """
    img_bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Cannot read: {input_path}")

    h, w = img_bgr.shape[:2]
    log.info(f"[Upscaler] {w}×{h}  →  {w * ESRGAN_SCALE}×{h * ESRGAN_SCALE}")

    t0 = time.time()
    out_bgr, _ = upsampler.enhance(img_bgr, outscale=ESRGAN_SCALE)
    elapsed = round(time.time() - t0, 2)

    oh, ow = out_bgr.shape[:2]
    log.info(f"[Upscaler] Output: {ow}×{oh}  in {elapsed}s")
    return out_bgr, elapsed


# ─────────────────────────────────────────────
#  QUALITY METRICS
# ─────────────────────────────────────────────
def compute_metrics(cloud_bgr: np.ndarray, upscaled_bgr: np.ndarray):
    """
    Print PSNR and SSIM comparing the AI upscale against a bicubic baseline.
    Higher PSNR/SSIM = upscale stayed structurally faithful to the source.
    """
    try:
        from skimage.metrics import peak_signal_noise_ratio as psnr
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        log.warning(
            "[Metrics] scikit-image not installed — skipping (pip install scikit-image)"
        )
        return

    h, w = upscaled_bgr.shape[:2]
    baseline = cv2.resize(cloud_bgr, (w, h), interpolation=cv2.INTER_CUBIC)
    base_rgb = cv2.cvtColor(baseline, cv2.COLOR_BGR2RGB)
    upscaled_rgb = cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2RGB)

    psnr_val = psnr(base_rgb, upscaled_rgb, data_range=255)
    ssim_val = ssim(base_rgb, upscaled_rgb, channel_axis=2, data_range=255)

    print(f"\n── Quality (RealESRGAN vs bicubic baseline) ──────────")
    print(f"  PSNR : {psnr_val:.2f} dB")
    print(f"  SSIM : {ssim_val:.4f}")
    print(f"──────────────────────────────────────────────────────")


# ─────────────────────────────────────────────
#  CLOUD CLIENT
# ─────────────────────────────────────────────
class Pipeline3Client:
    def __init__(self, cloud_url, client_id, save_dir):
        self.cloud_url = cloud_url.rstrip("/")
        self.client_id = client_id
        self.save_dir = Path(save_dir)
        self.session = requests.Session()
        self.session.headers["X-API-Key"] = API_KEY
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def health_check(self) -> bool:
        try:
            info = self.session.get(f"{self.cloud_url}/health", timeout=5).json()
            status = info.get("status", "unknown")
            loaded = info.get("model_loaded", False)
            log.info(
                f"[Cloud] {info.get('cloud_node')}  status={status}  "
                f"model={'ready' if loaded else 'loading'}  "
                f"device={info.get('device')}  "
                f"gpu_busy={info.get('gpu_busy')}  "
                f"queue={info.get('queue_depth')}"
            )
            if not loaded:
                log.warning("[Cloud] SD3 model still loading — wait and retry")
            return status == "healthy"
        except Exception as e:
            log.error(f"[Cloud] Unreachable: {e}")
            return False

    def show_queue_info(self):
        info = self.session.get(f"{self.cloud_url}/queue/info", timeout=5).json()
        q, j, s = info["queue"], info["jobs"], info["stats"]
        print(f"\n── Cloud dashboard ───────────────────────────────────")
        print(f"  Node    : {info['cloud_node']}  workers={info['workers']}")
        print(
            f"  Device  : {info.get('device', 'unknown')}  "
            f"gpu_busy={info.get('gpu_busy', '?')}"
        )
        print(f"  Queue   : {q['depth']}/{q['max_depth']}  ({q['fill_pct']}% full)")
        print(
            f"  Jobs    : queued={j['queued']}  processing={j['processing']}  "
            f"done={j['done']}  failed={j['failed']}"
        )
        print(f"  Avg gen : {s.get('avg_latency_sec', 'N/A')}s")
        print(f"──────────────────────────────────────────────────────")

    def submit_job(self, prompt, negative_prompt, steps, guidance) -> str:
        r = self.session.post(
            f"{self.cloud_url}/generate",
            json={
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "steps": steps,
                "guidance_scale": guidance,
                "client_id": self.client_id,
            },
            timeout=10,
        )

        if r.status_code == 429:
            raise RuntimeError("Rate limited by cloud")
        if r.status_code == 503:
            raise RuntimeError("Cloud queue full")
        if r.status_code != 202:
            raise RuntimeError(f"Submit failed [{r.status_code}]: {r.text}")

        d = r.json()
        log.info(
            f"[Cloud] Job accepted  id={d['job_id']}  "
            f"queue_pos={d.get('queue_position', '?')}"
        )
        return d["job_id"]

    def poll_until_done(self, job_id, poll_interval=DEFAULT_POLL_SEC) -> dict:
        spinner = ["|", "/", "─", "\\"]
        t0, tick = time.time(), 0
        while True:
            elapsed = time.time() - t0
            if elapsed > MAX_WAIT_SEC:
                raise TimeoutError(f"Timed out after {MAX_WAIT_SEC}s")

            job = self.session.get(
                f"{self.cloud_url}/status/{job_id}", timeout=10
            ).json()
            st = job["status"]

            if st == "QUEUED":
                print(
                    f"\r  {spinner[tick % 4]} QUEUED  pos={job.get('queue_position', '?')}  "
                    f"(+{elapsed:.0f}s) ",
                    end="",
                    flush=True,
                )
            elif st == "PROCESSING":
                print(
                    f"\r  {spinner[tick % 4]} PROCESSING  worker={job.get('worker_id')}  "
                    f"(+{elapsed:.0f}s) ",
                    end="",
                    flush=True,
                )
            elif st == "DONE":
                print()
                info = job["image_info"]
                log.info(
                    f"[Cloud] DONE  latency={job['latency_sec']}s  "
                    f"payload={info['payload_kb']} KB  "
                    f"upscale hint → {info['upscale_hint']['output_size']}"
                )
                return job
            elif st == "FAILED":
                print()
                raise RuntimeError(f"Cloud failed: {job.get('error')}")
            elif st == "CANCELLED":
                print()
                raise RuntimeError("Job cancelled")

            tick += 1
            time.sleep(poll_interval)

    def receive_cloud_image(self, job: dict) -> Path:
        """Decode base64 JPEG and save raw 512×512 cloud output."""
        jpeg_bytes = base64.b64decode(job["result_b64"])
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = job["prompt"][:20].replace(" ", "_").replace("/", "-")
        path = self.save_dir / f"cloud_raw_512_{ts}_{safe}.jpg"
        path.write_bytes(jpeg_bytes)
        log.info(f"[Client] Cloud raw → {path.name}  ({len(jpeg_bytes) / 1024:.1f} KB)")
        return path

    def run(self, prompt, negative_prompt, steps, guidance, poll_interval):
        print("=" * 57)
        print("  PIPELINE 3 — EDGE CLIENT")
        print("  Cloud SD3 512×512  →  RealESRGAN ×4  →  2048×2048")
        print("=" * 57)

        # Load upscaler first — fail fast before any network call
        log.info("[Client] Loading RealESRGAN weights...")
        upsampler = load_realesrgan()

        # Cloud handshake
        if not self.health_check():
            raise RuntimeError("Cloud node not healthy, aborting")
        self.show_queue_info()

        # Submit + poll
        t_wall = time.time()
        job_id = self.submit_job(prompt, negative_prompt, steps, guidance)
        job = self.poll_until_done(job_id, poll_interval)

        # Receive 512×512 JPEG from cloud
        raw_path = self.receive_cloud_image(job)
        cloud_bgr = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)

        # RealESRGAN upscale → 2048×2048
        log.info("[Client] Running RealESRGAN upscale...")
        upscaled_bgr, up_t = upscale(upsampler, raw_path)

        # Save final PNG
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = prompt[:20].replace(" ", "_").replace("/", "-")
        out_path = self.save_dir / f"final_2048_{ts}_{safe}.png"
        cv2.imwrite(str(out_path), upscaled_bgr)
        log.info(f"[Client] Final image → {out_path.name}")

        # Metrics
        compute_metrics(cloud_bgr, upscaled_bgr)

        # Summary
        total = round(time.time() - t_wall, 1)
        print(f"\n── Pipeline 3 complete ───────────────────────────────")
        print(f"  Prompt         : {prompt[:55]}")
        print(f"  Cloud gen      : {job['latency_sec']}s   (512×512 SD3 Medium)")
        print(f"  Payload        : {job['image_info']['payload_kb']} KB   (JPEG q=85)")
        print(f"  Upscale        : {up_t}s   (RealESRGAN x4plus)")
        print(f"  Final output   : 2048×2048 PNG")
        print(f"  Total time     : {total}s")
        print(f"  Raw cloud file : {raw_path.name}")
        print(f"  Final file     : {out_path.name}")
        print(f"──────────────────────────────────────────────────────")
        return out_path


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="Pipeline 3 — Cloud SD3 gen + Edge RealESRGAN upscale"
    )
    p.add_argument("--prompt", required=True)
    p.add_argument(
        "--negative", default="blurry, low quality, distorted, noisy, artifacts"
    )
    p.add_argument("--steps", type=int, default=28, help="SD3 inference steps (max 50)")
    p.add_argument("--guidance", type=float, default=7.0)
    p.add_argument("--cloud-url", default=DEFAULT_CLOUD_URL)
    p.add_argument("--client-id", default="edge-client-01")
    p.add_argument("--save-dir", default="./output")
    p.add_argument("--poll-interval", type=int, default=DEFAULT_POLL_SEC)
    p.add_argument(
        "--queue-info", action="store_true", help="Print cloud queue info and exit"
    )
    args = p.parse_args()

    client = Pipeline3Client(args.cloud_url, args.client_id, args.save_dir)

    if args.queue_info:
        if client.health_check():
            client.show_queue_info()
        return

    try:
        client.run(
            prompt=args.prompt,
            negative_prompt=args.negative,
            steps=args.steps,
            guidance=args.guidance,
            poll_interval=args.poll_interval,
        )
    except KeyboardInterrupt:
        print("\n[Client] Interrupted")
        sys.exit(0)
    except Exception as e:
        log.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()