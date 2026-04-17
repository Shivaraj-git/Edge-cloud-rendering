import warnings
# MUST be before ANY other imports
warnings.filterwarnings("ignore")

import cv2
import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips

# -----------------------------
# CONFIG
# -----------------------------
INPUT_PATH = "input/high_res.png"
LOW_RES_PATH = "output/cloud_output.png"
COMPRESSED_PATH = "output/compressed.jpg"
ENHANCED_PATH = "output/enhanced.png"
MODEL_PATH = "models/RealESRGAN_x4plus.pth"
 
# -----------------------------
# STEP 1: CLOUD SIMULATION
# -----------------------------
def cloud_render(input_path):
    print("[Cloud] Loading high-resolution image...")
    img = cv2.imread(input_path)

    # Downscale (simulate low-quality cloud rendering)
    print("[Cloud] Downscaling image...")
    
    #low res quality must be 1/4 of the org image dimension in the case of RealESRGAN_x4plus.pth
    h, w = img.shape[:2]
    scale = 4
    low_res = cv2.resize(img, (w // scale, h // scale))

    # Generate guidance (edges)
    print("[Cloud] Generating edge guidance...")
    gray = cv2.cvtColor(low_res, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    # Generate guidance (sobel gradient)
    print("[Cloud] Generating Sobel gradient guidance (NEW)...")
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)   # horizontal edges
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)   # vertical edges
    gradient_mag = np.sqrt(sobelx**2 + sobely**2)
    
    # Normalise to 0–255 so it can be saved as a standard image
    gradient_mag = cv2.normalize(
        gradient_mag, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
 
    # Save outputs
    cv2.imwrite(LOW_RES_PATH, low_res)
    cv2.imwrite("output/edges.png", edges)
    cv2.imwrite("output/gradient.png", gradient_mag)

    # Compression
    print("[Cloud] Compressing image...")
    cv2.imwrite(COMPRESSED_PATH, low_res, [int(cv2.IMWRITE_JPEG_QUALITY), 60])

    return low_res


# -----------------------------
# STEP 2: TRANSMISSION SIMULATION
# -----------------------------
def simulate_network():
    print("[Network] Simulating transmission delay...")
    time.sleep(0.1)  # simulate latency


# -----------------------------
# STEP 3: CLIENT AI ENHANCEMENT
# -----------------------------
def enhance_image():
    print("[Client] Loading compressed image...")
    
    img = cv2.imread(COMPRESSED_PATH, cv2.IMREAD_COLOR)
    
    # Sub Stage: Resolution Upscaling

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Client] Using device: {device}")

    # Model architecture (IMPORTANT)
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4
    )

    # Load Real-ESRGAN
    upsampler = RealESRGANer(
        scale=4,
        model_path=MODEL_PATH,
        model=model,
        tile=256,
        tile_pad=10,
        pre_pad=0,
        half=False  # set True if GPU
    )

    print("[Client] Enhancing image...")
    output, _ = upsampler.enhance(img, outscale=4)

    # Sub Stage: Guidance based Enhancement

    # Load guidance maps transmitted from cloud
    edges    = cv2.imread("output/edges.png",    0)  # Canny  (grayscale)
    gradient = cv2.imread("output/gradient.png", 0)  # Sobel  (grayscale)

    # Resize both maps to match the ESRGAN output resolution
    edges    = cv2.resize(edges,    (output.shape[1], output.shape[0]))
    gradient = cv2.resize(gradient, (output.shape[1], output.shape[0]))

    # Combine: Canny gives hard structure, Sobel gives soft texture
    # 0.6 / 0.4 weighting found empirically — adjust to taste
    combined_guidance = cv2.addWeighted(edges, 0.6, gradient, 0.4, 0)

    # Normalise combined map to [0, 1]
    combined_guidance = combined_guidance.astype(np.float32) / 255.0

    # Expand to 3 channels so it can multiply with the colour image
    guidance_3ch = np.stack([combined_guidance] * 3, axis=2)

    # Compute detail layer via unsharp mask
    blurred      = cv2.GaussianBlur(output.astype(np.float32), (0, 0), sigmaX=2)
    detail_layer = output.astype(np.float32) - blurred

    # Adaptive alpha — regions with more guidance signal get stronger
    edge_strength = np.mean(combined_guidance)
    alpha = np.clip(edge_strength * 2, 0.1, 0.5)
    print(f"[Client] Adaptive guidance strength (alpha): {alpha:.3f}")

    # Apply guidance
    sharpened = output.astype(np.float32) + alpha * guidance_3ch * detail_layer
    final = np.clip(sharpened, 0, 255).astype(np.uint8)

    cv2.imwrite(ENHANCED_PATH, final)
    print(f"[Client] Enhanced image saved to: {ENHANCED_PATH}\n")
 
    return final


# -----------------------------
# STEP 4: VISUALIZATION
# -----------------------------
def show_results(original, low_res, enhanced):
    print("[Visualization] Displaying results...")

    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    low_res = cv2.cvtColor(low_res, cv2.COLOR_BGR2RGB)
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Original (High Quality)")
    plt.imshow(original)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Cloud Output (Low Quality)")
    plt.imshow(low_res)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Client Enhanced (AI Output)")
    plt.imshow(enhanced)
    plt.axis('off')

    plt.show()

# -----------------------------
# METRICS CALCULATION
# -----------------------------

def calculate_metrics(original, enhanced):
    print("\n=== QUALITY METRICS ===")

    # Convert images to same format
    enhanced_np = enhanced

    # # Resize enhanced to match original (important!)
    enhanced_np = cv2.resize(enhanced_np, (original.shape[1], original.shape[0]))

    # Convert BGR → RGB
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    enhanced_rgb = cv2.cvtColor(enhanced_np, cv2.COLOR_BGR2RGB)

    # -------------------------
    # PSNR
    # -------------------------
    psnr_value = psnr(original_rgb, enhanced_rgb)
    print(f"PSNR: {psnr_value:.2f} dB")

    # -------------------------
    # SSIM
    # -------------------------
    ssim_value = ssim(original_rgb, enhanced_rgb, channel_axis=2)
    print(f"SSIM: {ssim_value:.4f}")

    # -------------------------
    # LPIPS (Deep Learning Metric)
    # -------------------------
    loss_fn = lpips.LPIPS(net='alex')  # pretrained model

    # Convert to tensor (range -1 to 1)
    def to_tensor(img):
        img = cv2.resize(img, (256, 256))
        img = img.astype(np.float32) / 255.0
        img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)
        return img * 2 - 1

    t1 = to_tensor(original_rgb)
    t2 = to_tensor(enhanced_rgb) 

    lpips_value = loss_fn(t1, t2)
    print(f"LPIPS: {lpips_value.item():.4f}")

    print("========================\n")

 
# -----------------------------
# MAIN PIPELINE
# -----------------------------
def main():
    if not os.path.exists("output"):
        os.makedirs("output")

    print("=== EDGE-CLOUD RENDERING PIPELINE START ===")

    # Original image
    original = cv2.imread(INPUT_PATH)

    # Step 1: Cloud
    low_res = cloud_render(INPUT_PATH)

    # Step 2: Network
    simulate_network()

    # Step 3: Client
    enhanced = enhance_image()

    # Metrics
    calculate_metrics(original, enhanced)

    # Step 4: Show results
    show_results(original, low_res, enhanced)

    print("=== PROCESS COMPLETE ===")


if __name__ == "__main__":
    main()