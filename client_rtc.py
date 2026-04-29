"""
=============================================================
PIPELINE 3 — EDGE CLIENT  (WebRTC + Edge-Guided Upscaling)
Cloud SD3 512×512  →  RealESRGAN x4  →  2048×2048
=============================================================
What's new vs v1:

  ┌─ WebRTC delivery ───────────────────────────────────────┐
  │ Client opens a WebRTC data channel to the cloud before  │
  │ submitting the job. The cloud PUSHES the result the     │
  │ moment inference completes — no polling round trips.    │
  │ HTTP polling is kept as a transparent fallback.         │
  └─────────────────────────────────────────────────────────┘
  ┌─ Decompression ─────────────────────────────────────────┐
  │ Cloud now sends JPEG q=75 + zlib. Client decompresses   │
  │ via zlib → JPEG decode before handing off to ESRGAN.   │
  └─────────────────────────────────────────────────────────┘
  ┌─ Edge-guided upscaling ─────────────────────────────────┐
  │ Cloud sends three extra maps computed from the 512×512: │
  │   • edge  — Sobel magnitude (where edges are)           │
  │   • gx/gy — signed gradient components (edge direction) │
  │ After RealESRGAN upscale the client:                    │
  │   1. Upscales all three maps to 2048×2048 (Lanczos)    │
  │   2. Computes gradient direction alignment between      │
  │      the cloud map and the local ESRGAN gradient        │
  │   3. Applies unsharp-mask detail weighted by            │
  │      edge_strength × gradient_alignment                 │
  │ → Sharp edges, smooth areas left natural.               │
  └─────────────────────────────────────────────────────────┘

Install (adds to existing requirements):
  pip install aiortc

Run:
  export CLOUD_API_KEY=pipeline3-cloud-key
  python edge1.py --prompt "a futuristic city at sunset"
=============================================================
"""

import argparse
import asyncio
import base64
import json
import logging
import os
import sys
import time
import zlib
from datetime import datetime
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
from PIL import Image

# ─────────────────────────────────────────────
# OPTIONAL: aiortc for WebRTC
# ─────────────────────────────────────────────
try:
    from aiortc import (
        RTCConfiguration,
        RTCIceServer,
        RTCPeerConnection,
        RTCSessionDescription,
    )

    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DEFAULT_CLOUD_URL = "http://localhost:8765"
API_KEY = os.environ.get("CLOUD_API_KEY", "pipeline3-cloud-key")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "RealESRGAN_x4plus.pth")
ESRGAN_SCALE = 4
TILE_SIZE = 256
MAX_WAIT_SEC = 600
DEFAULT_POLL_SEC = 3

STUN_SERVERS = [
    "stun:stun.l.google.com:19302",
    "stun:stun1.l.google.com:19302",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("EdgeClient")


# ─────────────────────────────────────────────
# REALESRGAN LOADER
# ─────────────────────────────────────────────
def load_realesrgan():
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise RuntimeError(f"RealESRGAN import FAILED: {e}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Weights not found at: {MODEL_PATH}\n"
            "  mkdir -p models\n"
            "  wget -O models/RealESRGAN_x4plus.pth \\\n"
            "    https://github.com/xinntao/Real-ESRGAN/releases/download/"
            "v0.1.0/RealESRGAN_x4plus.pth"
        )

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"[Upscaler] Device: {dev}")

    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

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
        half=torch.cuda.is_available(),
        device=dev,
    )
    log.info(f"[Upscaler] RealESRGAN x{ESRGAN_SCALE} ready")
    return upsampler


# ─────────────────────────────────────────────
# DECOMPRESSION HELPERS
# ─────────────────────────────────────────────
def decode_image(result_b64: str, compression: str = "zlib+jpeg") -> np.ndarray:
    """
    Decode cloud image payload → BGR numpy array.

    compression='zlib+jpeg' (default, v2):
        base64 → zlib decompress → JPEG decode
    compression='jpeg' (legacy v1 HTTP fallback):
        base64 → JPEG decode
    """
    raw = base64.b64decode(result_b64)
    if "zlib" in compression:
        jpeg_bytes = zlib.decompress(raw)
    else:
        jpeg_bytes = raw
    img_pil = Image.open(BytesIO(jpeg_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def decode_edge_data(edge_data: dict) -> tuple:
    """
    Decode the three cloud edge/gradient maps.

    Each is stored as:  base64 → zlib decompress → PNG → uint8 numpy

    Returns:
        edge_u8  : (H, W) uint8 — Sobel magnitude, normalised 0-255
        gx_u8    : (H, W) uint8 — X-gradient; 128 = zero gradient
        gy_u8    : (H, W) uint8 — Y-gradient; 128 = zero gradient
    """

    def _dec(b64_str: str) -> np.ndarray:
        compressed = base64.b64decode(b64_str)
        png_bytes = zlib.decompress(compressed)
        return np.array(Image.open(BytesIO(png_bytes)))

    return _dec(edge_data["edge"]), _dec(edge_data["gx"]), _dec(edge_data["gy"])


# ─────────────────────────────────────────────
# EDGE-GUIDED UPSCALE ENHANCEMENT
# ─────────────────────────────────────────────
def edge_guided_enhance(
    upscaled_bgr: np.ndarray,
    edge_u8: np.ndarray,
    gx_u8: np.ndarray,
    gy_u8: np.ndarray,
    gx_scale: float = 4.0,
    strength: float = 1.2,
) -> np.ndarray:
    """
    Selectively sharpen the RealESRGAN output using the Sobel edge and
    gradient maps that the cloud computed from the original 512×512.

    Why this works better than a plain unsharp mask:
      • 'edge_u8' tells us WHERE genuine structural edges are in the
        original image — not ESRGAN artefacts or noise.
      • 'gx/gy_u8' encode the DIRECTION of those edges, so we can
        compare them with the local gradient of the upscaled image.
        Only detail that is aligned with the original edge direction
        is amplified; misaligned sharpening (ringing, halos) is
        suppressed via the alignment weight.

    Algorithm:
      1. Upscale all three maps to 2048×2048 with Lanczos (continuous).
      2. Reconstruct signed Gx/Gy:  gx_f = (u8 - 128) * gx_scale
      3. Compute per-pixel gradient direction from cloud maps.
      4. Compute local gradient direction from ESRGAN output.
      5. Alignment weight = cosine similarity (clamped to [0,1]).
      6. Detail layer = ESRGAN - GaussianBlur(ESRGAN).
      7. Enhanced = ESRGAN + strength * detail * edge_mask * alignment

    Args:
        upscaled_bgr : RealESRGAN output, uint8 BGR, shape (2048, 2048, 3)
        edge_u8      : Sobel magnitude map from cloud, uint8, (512, 512)
        gx_u8/gy_u8  : Gradient maps from cloud, uint8, (512, 512)
        gx_scale     : Encoding divisor used by cloud (default 4.0)
        strength     : Sharpening intensity (0 = off, 2 = aggressive)
    """
    H, W = upscaled_bgr.shape[:2]

    # ── Step 1: Upscale cloud maps to output resolution ────────────
    edge_f = cv2.resize(
        edge_u8.astype(np.float32), (W, H), interpolation=cv2.INTER_LANCZOS4
    )
    gx_u8_big = cv2.resize(gx_u8, (W, H), interpolation=cv2.INTER_LANCZOS4)
    gy_u8_big = cv2.resize(gy_u8, (W, H), interpolation=cv2.INTER_LANCZOS4)

    # ── Step 2: Reconstruct signed gradients ──────────────────────
    gx_cloud = (gx_u8_big.astype(np.float32) - 128.0) * gx_scale
    gy_cloud = (gy_u8_big.astype(np.float32) - 128.0) * gx_scale

    # ── Step 3: Cloud gradient direction (unit vectors) ───────────
    cloud_mag = np.sqrt(gx_cloud**2 + gy_cloud**2).clip(1e-6)
    gx_dir = gx_cloud / cloud_mag  # unit X
    gy_dir = gy_cloud / cloud_mag  # unit Y

    # ── Step 4: Local gradient direction from ESRGAN output ───────
    gray_up = cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dx = cv2.Sobel(gray_up, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray_up, cv2.CV_32F, 0, 1, ksize=3)
    local_mag = np.sqrt(dx**2 + dy**2).clip(1e-6)
    dx_dir = dx / local_mag
    dy_dir = dy / local_mag

    # ── Step 5: Gradient alignment weight ─────────────────────────
    # Dot product of cloud vs local gradient unit vectors → 0-1
    alignment = (dx_dir * gx_dir + dy_dir * gy_dir).clip(0.0, 1.0)  # (H, W)

    # ── Step 6: Detail via unsharp mask ───────────────────────────
    img_f = upscaled_bgr.astype(np.float32)
    blurred = cv2.GaussianBlur(img_f, (5, 5), 1.0)
    detail = img_f - blurred  # high-frequency residual, shape (H, W, 3)

    # ── Step 7: Apply edge-guided sharpening ──────────────────────
    # Combine: strong edge AND aligned direction → full sharpening
    edge_norm = (edge_f / 255.0)[..., np.newaxis]  # (H, W, 1)
    alignment_3ch = alignment[..., np.newaxis]  # (H, W, 1)
    weight = edge_norm * alignment_3ch  # (H, W, 1)

    enhanced = img_f + strength * detail * weight
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

    mean_edge = float(edge_norm.mean())
    mean_align = float(alignment.mean())
    log.info(
        f"[EdgeGuide] Sharpening applied — "
        f"mean edge={mean_edge:.3f}  mean align={mean_align:.3f}  strength={strength}"
    )
    return enhanced


# ─────────────────────────────────────────────
# UPSCALE (RealESRGAN)
# ─────────────────────────────────────────────
def upscale(upsampler, cloud_bgr: np.ndarray) -> tuple:
    h, w = cloud_bgr.shape[:2]
    log.info(f"[Upscaler] {w}×{h} → {w * ESRGAN_SCALE}×{h * ESRGAN_SCALE}")
    t0 = time.time()
    out_bgr, _ = upsampler.enhance(cloud_bgr, outscale=ESRGAN_SCALE)
    elapsed = round(time.time() - t0, 2)
    oh, ow = out_bgr.shape[:2]
    log.info(f"[Upscaler] Output: {ow}×{oh} in {elapsed}s")
    return out_bgr, elapsed


# ─────────────────────────────────────────────
# QUALITY METRICS
# ─────────────────────────────────────────────
def compute_metrics(cloud_bgr: np.ndarray, upscaled_bgr: np.ndarray):
    try:
        from skimage.metrics import peak_signal_noise_ratio as psnr
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        log.warning("[Metrics] scikit-image not installed — skipping")
        return
    h, w = upscaled_bgr.shape[:2]
    baseline = cv2.resize(cloud_bgr, (w, h), interpolation=cv2.INTER_CUBIC)
    base_rgb = cv2.cvtColor(baseline, cv2.COLOR_BGR2RGB)
    up_rgb = cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2RGB)
    psnr_val = psnr(base_rgb, up_rgb, data_range=255)
    ssim_val = ssim(base_rgb, up_rgb, channel_axis=2, data_range=255)
    print(f"\n── Quality (RealESRGAN+EdgeGuide vs bicubic baseline) ──")
    print(f" PSNR : {psnr_val:.2f} dB")
    print(f" SSIM : {ssim_val:.4f}")
    print(f"────────────────────────────────────────────────────────")


# ─────────────────────────────────────────────
# WEBRTC CONNECTION
# ─────────────────────────────────────────────
async def establish_webrtc(cloud_url: str, client_id: str) -> tuple:
    """
    Create a WebRTC peer connection and data channel to the cloud.

    Flow:
      1. Client creates RTCPeerConnection + data channel "results"
      2. Client creates SDP offer (includes data channel description)
      3. Client HTTP POSTs offer to cloud /webrtc/offer
      4. Cloud creates peer connection, sets up ondatachannel handler,
         returns SDP answer
      5. Client sets remote description → ICE negotiation starts
      6. When ICE completes the cloud's ondatachannel fires, channel opens
      7. Cloud can now push() results at any time

    Returns:
        pc             : RTCPeerConnection (keep alive until done)
        result_future  : asyncio.Future that resolves with the WebRTC payload
    """
    cfg = RTCConfiguration(iceServers=[RTCIceServer(urls=STUN_SERVERS)])
    pc = RTCPeerConnection(configuration=cfg)
    ch = pc.createDataChannel("results", ordered=True)

    loop = asyncio.get_event_loop()
    result_future = loop.create_future()

    @ch.on("open")
    def on_open():
        log.info("[WebRTC] Data channel open — ready to receive cloud result")

    @ch.on("message")
    def on_message(message):
        # message can be str or bytes
        if isinstance(message, (bytes, bytearray)):
            message = message.decode("utf-8")
        try:
            data = json.loads(message)
        except json.JSONDecodeError as e:
            if not result_future.done():
                result_future.set_exception(RuntimeError(f"Bad JSON from cloud: {e}"))
            return
        if data.get("type") == "result" and not result_future.done():
            size_kb = len(message) // 1024
            log.info(f"[WebRTC] ◀ Result received ({size_kb} KB JSON)")
            result_future.set_result(data)

    @ch.on("close")
    def on_close():
        if not result_future.done():
            result_future.set_exception(
                RuntimeError("WebRTC channel closed before result")
            )

    @pc.on("connectionstatechange")
    async def on_state():
        state = pc.connectionState
        log.info(f"[WebRTC] Connection state: {state}")
        if state == "failed" and not result_future.done():
            result_future.set_exception(RuntimeError("WebRTC connection failed"))

    # Create and send SDP offer
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    resp = requests.post(
        f"{cloud_url}/webrtc/offer",
        json={
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
            "client_id": client_id,
        },
        headers={"X-API-Key": API_KEY, "Content-Type": "application/json"},
        timeout=15,
    )
    resp.raise_for_status()
    answer_data = resp.json()
    answer = RTCSessionDescription(sdp=answer_data["sdp"], type=answer_data["type"])
    await pc.setRemoteDescription(answer)
    log.info("[WebRTC] SDP exchange complete — ICE negotiating...")

    return pc, result_future


# ─────────────────────────────────────────────
# HTTP CLIENT  (submit + fallback polling)
# ─────────────────────────────────────────────
class CloudHTTPClient:
    def __init__(self, cloud_url: str, client_id: str):
        self.cloud_url = cloud_url.rstrip("/")
        self.client_id = client_id
        self.session = requests.Session()
        self.session.headers["X-API-Key"] = API_KEY

    def health_check(self) -> bool:
        try:
            info = self.session.get(f"{self.cloud_url}/health", timeout=5).json()
            status = info.get("status", "unknown")
            loaded = info.get("model_loaded", False)
            log.info(
                f"[Cloud] {info.get('cloud_node')} status={status} "
                f"model={'ready' if loaded else 'loading'} "
                f"device={info.get('device')} "
                f"webrtc={info.get('webrtc_available', False)} "
                f"queue={info.get('queue_depth')}"
            )
            if not loaded:
                log.warning("[Cloud] SD3 still loading — retry in a few seconds")
            return status == "healthy"
        except Exception as e:
            log.error(f"[Cloud] Unreachable: {e}")
            return False

    def show_queue_info(self):
        info = self.session.get(f"{self.cloud_url}/queue/info", timeout=5).json()
        q, j, s = info["queue"], info["jobs"], info["stats"]
        print(f"\n── Cloud dashboard ────────────────────────────────────────")
        print(f" Node        : {info['cloud_node']}  workers={info['workers']}")
        print(f" Device      : {info.get('device')}  gpu_busy={info.get('gpu_busy')}")
        print(f" WebRTC      : {info.get('webrtc', {})}")
        print(f" Compression : {info.get('compression')}")
        print(f" Queue       : {q['depth']}/{q['max_depth']}  ({q['fill_pct']}% full)")
        print(
            f" Jobs        : queued={j['queued']}  processing={j['processing']}  done={j['done']}"
        )
        print(f" Avg gen     : {s.get('avg_latency_sec', 'N/A')}s")
        print(f"───────────────────────────────────────────────────────────")

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
            f"[Cloud] Job accepted  id={d['job_id']}  pos={d.get('queue_position', '?')}"
        )
        return d["job_id"]

    def http_poll_until_done(
        self, job_id: str, poll_interval: int = DEFAULT_POLL_SEC
    ) -> dict:
        """HTTP polling fallback — used when WebRTC is unavailable or times out."""
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
                    f"\r {spinner[tick % 4]} QUEUED pos={job.get('queue_position', '?')} "
                    f"(+{elapsed:.0f}s) ",
                    end="",
                    flush=True,
                )
            elif st == "PROCESSING":
                print(
                    f"\r {spinner[tick % 4]} PROCESSING worker={job.get('worker_id')} "
                    f"(+{elapsed:.0f}s) ",
                    end="",
                    flush=True,
                )
            elif st == "DONE":
                print()
                info = job.get("image_info", {})
                log.info(
                    f"[Cloud] DONE latency={job['latency_sec']}s "
                    f"payload={info.get('payload_kb')} KB "
                    f"compression={job.get('compression', 'jpeg')}"
                )
                return job
            elif st == "FAILED":
                print()
                raise RuntimeError(f"Cloud failed: {job.get('error')}")
            elif st == "CANCELLED":
                print()
                raise RuntimeError("Job was cancelled")
            tick += 1
            time.sleep(poll_interval)


# ─────────────────────────────────────────────
# MAIN PIPELINE  (async)
# ─────────────────────────────────────────────
async def run_pipeline(args, save_dir: Path) -> Path:
    print("=" * 62)
    print(" PIPELINE 3 — EDGE CLIENT")
    print(" WebRTC delivery + zlib decompression + edge-guided upscale")
    print(f" Cloud SD3 {512}×{512} → RealESRGAN ×4 + EdgeGuide → {2048}×{2048}")
    print("=" * 62)

    # ── Load RealESRGAN first — fail fast before any network call ──
    log.info("[Client] Loading RealESRGAN weights...")
    upsampler = load_realesrgan()

    http_client = CloudHTTPClient(args.cloud_url, args.client_id)
    if not http_client.health_check():
        raise RuntimeError("Cloud node not healthy, aborting")
    http_client.show_queue_info()

    t_wall = time.time()

    # ── Establish WebRTC connection ────────────────────────────────
    pc, result_future = None, None
    if WEBRTC_AVAILABLE:
        log.info("[WebRTC] Establishing data channel with cloud...")
        try:
            pc, result_future = await establish_webrtc(args.cloud_url, args.client_id)
        except Exception as e:
            log.warning(f"[WebRTC] Setup failed ({e}) — will use HTTP polling")
            pc, result_future = None, None
    else:
        log.warning(
            "[WebRTC] aiortc not installed — using HTTP polling (pip install aiortc)"
        )

    # ── Submit generation job ──────────────────────────────────────
    job_id = http_client.submit_job(
        args.prompt, args.negative, args.steps, args.guidance
    )

    # ── Receive result ─────────────────────────────────────────────
    webrtc_payload = None
    http_job = None

    if result_future is not None:
        log.info(f"[WebRTC] Waiting for cloud push  (job {job_id[:8]})...")
        try:
            webrtc_payload = await asyncio.wait_for(result_future, timeout=MAX_WAIT_SEC)
        except asyncio.TimeoutError:
            log.warning("[WebRTC] Push timed out — falling back to HTTP polling")
        except Exception as e:
            log.warning(f"[WebRTC] Receive error ({e}) — falling back to HTTP polling")
        finally:
            if pc:
                await pc.close()

    if webrtc_payload is None:
        log.info("[Client] HTTP polling fallback...")
        http_job = http_client.http_poll_until_done(job_id, args.poll_interval)

    # ── Decode compressed image ────────────────────────────────────
    if webrtc_payload is not None:
        compression = webrtc_payload.get("compression", "zlib+jpeg")
        cloud_bgr = decode_image(webrtc_payload["image_zlib_b64"], compression)
        edge_data = webrtc_payload.get("edge_data")
        meta = webrtc_payload.get("metadata", {})
        payload_kb = meta.get("payload_kb", "?")
        latency_sec = meta.get("latency_sec", "?")
        delivery = "WebRTC push"
        log.info(
            f"[Cloud] Result decoded  latency={latency_sec}s  payload={payload_kb} KB"
        )
    else:
        compression = http_job.get("compression", "jpeg")
        cloud_bgr = decode_image(http_job["result_b64"], compression)
        edge_data = http_job.get("edge_data")
        payload_kb = http_job.get("image_info", {}).get("payload_kb", "?")
        latency_sec = http_job.get("latency_sec", "?")
        delivery = "HTTP poll"

    # Save raw 512×512 cloud output for reference
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = args.prompt[:20].replace(" ", "_").replace("/", "-")
    raw_path = save_dir / f"cloud_raw_512_{ts}_{safe_name}.jpg"
    cv2.imwrite(str(raw_path), cloud_bgr)
    log.info(f"[Client] Cloud raw saved → {raw_path.name}")

    # ── RealESRGAN upscale 512 → 2048 ─────────────────────────────
    log.info("[Client] Running RealESRGAN upscale...")
    upscaled_bgr, up_t = upscale(upsampler, cloud_bgr)

    # ── Edge-guided sharpening ─────────────────────────────────────
    if edge_data is not None:
        log.info("[EdgeGuide] Decoding cloud Sobel / gradient maps...")
        edge_u8, gx_u8, gy_u8 = decode_edge_data(edge_data)
        gx_scale = float(edge_data.get("gx_scale", 4.0))
        log.info("[EdgeGuide] Applying directional edge-guided sharpening...")
        upscaled_bgr = edge_guided_enhance(
            upscaled_bgr, edge_u8, gx_u8, gy_u8, gx_scale=gx_scale, strength=1.2
        )
    else:
        log.warning("[EdgeGuide] No edge data received — skipping (cloud may be v1)")

    # ── Save final 2048×2048 PNG ───────────────────────────────────
    out_path = save_dir / f"final_2048_{ts}_{safe_name}.png"
    cv2.imwrite(str(out_path), upscaled_bgr)
    log.info(f"[Client] Final image saved → {out_path.name}")

    # ── Quality metrics ────────────────────────────────────────────
    compute_metrics(cloud_bgr, upscaled_bgr)

    # ── Summary ───────────────────────────────────────────────────
    total = round(time.time() - t_wall, 1)
    print(f"\n── Pipeline 3 complete ──────────────────────────────────────")
    print(f" Prompt      : {args.prompt[:55]}")
    print(f" Delivery    : {delivery}")
    print(f" Cloud gen   : {latency_sec}s  (512×512 SD3 Medium)")
    print(f" Payload     : {payload_kb} KB  (JPEG q=75 + zlib)")
    print(
        f" Edge data   : {'Sobel magnitude + Gx/Gy (cloud-computed)' if edge_data else 'absent'}"
    )
    print(f" Upscale     : {up_t}s  (RealESRGAN x4plus)")
    print(f" EdgeGuide   : {'applied' if edge_data else 'skipped'}")
    print(f" Final       : 2048×2048 PNG")
    print(f" Total time  : {total}s")
    print(f" Raw file    : {raw_path.name}")
    print(f" Final file  : {out_path.name}")
    print(f"─────────────────────────────────────────────────────────────")

    return out_path


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description=(
            "Pipeline 3 — Cloud SD3 gen + WebRTC delivery "
            "+ edge-guided RealESRGAN upscale"
        )
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
        "--queue-info", action="store_true", help="Print cloud dashboard and exit"
    )
    args = p.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.queue_info:
        c = CloudHTTPClient(args.cloud_url, args.client_id)
        if c.health_check():
            c.show_queue_info()
        return

    try:
        asyncio.run(run_pipeline(args, save_dir))
    except KeyboardInterrupt:
        print("\n[Client] Interrupted")
        sys.exit(0)
    except Exception as e:
        log.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
