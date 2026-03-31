import argparse
import time
import numpy as np
from PIL import Image, ImageFilter
from pathlib import Path
from scipy.cluster.vq import kmeans2

# =====================================================================
# CIELAB COLOR SPACE CONVERSIONS (Vectorized for extreme speed)
# RGB distance is terrible for human perception. CIELAB distance 
# perfectly maps to how humans see color differences.
# =====================================================================
def rgb_to_lab(rgb_array):
    rgb = rgb_array.astype(float) / 255.0
    
    mask = rgb > 0.04045
    rgb[mask] = ((rgb[mask] + 0.055) / 1.055) ** 2.4
    rgb[~mask] = rgb[~mask] / 12.92
    rgb *= 100.0

    xyz = np.zeros_like(rgb)
    xyz[:, 0] = rgb[:, 0] * 0.4124564 + rgb[:, 1] * 0.3575761 + rgb[:, 2] * 0.1804375
    xyz[:, 1] = rgb[:, 0] * 0.2126729 + rgb[:, 1] * 0.7151522 + rgb[:, 2] * 0.0721750
    xyz[:, 2] = rgb[:, 0] * 0.0193339 + rgb[:, 1] * 0.1191920 + rgb[:, 2] * 0.9503041

    xyz_norm = xyz / np.array([95.047, 100.0, 108.883])
    mask = xyz_norm > 0.008856
    
    f_xyz = np.zeros_like(xyz_norm)
    f_xyz[mask] = np.cbrt(xyz_norm[mask])
    f_xyz[~mask] = (7.787 * xyz_norm[~mask]) + (16.0 / 116.0)

    lab = np.zeros_like(rgb)
    lab[:, 0] = (116.0 * f_xyz[:, 1]) - 16.0
    lab[:, 1] = 500.0 * (f_xyz[:, 0] - f_xyz[:, 1])
    lab[:, 2] = 200.0 * (f_xyz[:, 1] - f_xyz[:, 2])
    return lab

def lab_to_rgb(lab_array):
    lab = lab_array.astype(float)
    y = (lab[:, 0] + 16.0) / 116.0
    x = lab[:, 1] / 500.0 + y
    z = y - lab[:, 2] / 200.0
    
    xyz = np.column_stack((x, y, z))
    mask = xyz > 0.2068966
    xyz[mask] = xyz[mask] ** 3.0
    xyz[~mask] = (xyz[~mask] - 16.0 / 116.0) / 7.787
    
    xyz *= np.array([95.047, 100.0, 108.883]) / 100.0
    
    rgb = np.zeros_like(xyz)
    rgb[:, 0] = xyz[:, 0] * 3.2404542 - xyz[:, 1] * 1.5371385 - xyz[:, 2] * 0.4985314
    rgb[:, 1] = -xyz[:, 0] * 0.9692660 + xyz[:, 1] * 1.8760108 + xyz[:, 2] * 0.0415560
    rgb[:, 2] = xyz[:, 0] * 0.0556434 - xyz[:, 1] * 0.2040259 + xyz[:, 2] * 1.0572252
    
    mask = rgb > 0.0031308
    rgb[mask] = 1.055 * (rgb[mask] ** (1 / 2.4)) - 0.055
    rgb[~mask] = rgb[~mask] * 12.92
    
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

# =====================================================================
# SALIENCY-WEIGHTED SAMPLING (Anti-Gravity Trick)
# =====================================================================
def get_biased_samples(img, sample_size=5000, edge_weight=10.0):
    """
    Downsamples the image, detects edges, and uses those edges as a 
    probability map. A tiny torch on a black wall generates massive edge 
    values, guaranteeing its pixels are over-sampled and preserved!
    """
    # Downscale heavily to simulate realtime performance constraints
    analysis_img = img.resize((128, 128), resample=Image.Resampling.BILINEAR)
    
    # Simple, fast edge detection
    edges = analysis_img.convert('L').filter(ImageFilter.FIND_EDGES)
    
    pixels = np.array(analysis_img).reshape(-1, 3)
    edge_vals = np.array(edges).reshape(-1)
    
    # Base probability = 1.0 (so flat background walls still get sampled a bit)
    # Edge pixels get multiplied by the edge_weight parameter.
    weights = 1.0 + (edge_vals.astype(float) / 255.0) * edge_weight
    probs = weights / np.sum(weights)
    
    # Fast numpy random choice based on our saliency map
    actual_sample_size = min(sample_size, len(pixels))
    indices = np.random.choice(len(pixels), size=actual_sample_size, p=probs, replace=False)
    
    return pixels[indices]

# =====================================================================
# CORE QUANTIZATION PIPELINE
# =====================================================================
def extract_palette(img, num_colors, prev_palette_lab=None, edge_weight=10.0, samples=5000):
    # 1. Grab our anti-gravity weighted pixel samples
    sampled_rgb = get_biased_samples(img, samples, edge_weight)
    
    # 2. Convert to human-perception color space
    sampled_lab = rgb_to_lab(sampled_rgb)
    
    # 3. K-MEANS CLUSTERING WITH TEMPORAL SEEDING
    if prev_palette_lab is not None and len(prev_palette_lab) == num_colors:
        # REALTIME HACK: Seed K-means with the previous frame's palette!
        # Because we are starting 95% close to the answer, we only need 3 iterations.
        centroids, _ = kmeans2(sampled_lab, prev_palette_lab, iter=3, minit='matrix')
    else:
        # First frame: Use robust K-Means++ initialization and allow more iterations
        try:
            centroids, _ = kmeans2(sampled_lab, num_colors, iter=15, minit='++')
        except ValueError:
            # Fallback for older SciPy versions
            centroids, _ = kmeans2(sampled_lab, num_colors, iter=15, minit='points')
            
    return centroids

def sort_by_luminance(palette_rgb):
    luminance = 0.299 * palette_rgb[:, 0] + 0.587 * palette_rgb[:, 1] + 0.114 * palette_rgb[:, 2]
    return palette_rgb[np.argsort(luminance)]

# =====================================================================
# DITHERING
# =====================================================================
def dither_pixelated(img, palette_rgb, scale_factor):
    original_width, original_height = img.size
    scale_factor = max(1, scale_factor)
    small_width, small_height = max(1, original_width // scale_factor), max(1, original_height // scale_factor)
    
    img_small = img.resize((small_width, small_height), resample=Image.Resampling.BOX)
    
    flat_palette = palette_rgb.flatten().tolist()
    flat_palette += [0] * (768 - len(flat_palette))
    
    palette_img = Image.new('P', (1, 1))
    palette_img.putpalette(flat_palette[:768])
    
    dithered_small = img_small.quantize(palette=palette_img, dither=Image.Dither.FLOYDSTEINBERG)
    return dithered_small.resize((original_width, original_height), resample=Image.Resampling.NEAREST)

# =====================================================================
# BATCH PROCESSOR
# =====================================================================
def process_video_frames(input_dir, output_dir, num_colors, scale_factor, blend_factor, edge_weight, samples):
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    files_to_process = sorted([f for f in in_path.iterdir() if f.suffix.lower() in {'.png', '.jpg', '.jpeg'}])
    total_files = len(files_to_process)
    if total_files == 0: return

    # State tracking for EMA and K-Means Seeding
    prev_palette_rgb = None
    prev_palette_lab = None

    print(f"Starting Saliency-Weighted CIELAB Pipeline on {total_files} frames...")

    for i, img_file in enumerate(files_to_process, 1):
        t0 = time.time()
        try:
            img = Image.open(img_file).convert('RGB')
            
            # 1. Extract Target Palette in LAB Space
            target_palette_lab = extract_palette(img, num_colors, prev_palette_lab, edge_weight, samples)
            
            # 2. Convert to RGB and Sort by Luminance (per user request)
            target_palette_rgb = lab_to_rgb(target_palette_lab)
            target_palette_rgb = sort_by_luminance(target_palette_rgb)
            
            # 3. EMA TEMPORAL BLEND
            if prev_palette_rgb is not None:
                final_palette_rgb = ((1.0 - blend_factor) * prev_palette_rgb) + (blend_factor * target_palette_rgb)
            else:
                final_palette_rgb = target_palette_rgb
                
            # Update States
            prev_palette_rgb = final_palette_rgb
            # Re-convert blended RGB back to LAB to act as the exact seed for next frame
            prev_palette_lab = rgb_to_lab(final_palette_rgb) 
            
            # 4. Final Processing
            final_palette_uint8 = np.clip(final_palette_rgb, 0, 255).astype(np.uint8)
            final_img = dither_pixelated(img, final_palette_uint8, scale_factor)
            final_img.save(out_path / f"{img_file.stem}-dither.png")
            
            elapsed = time.time() - t0
            print(f"\r[{i}/{total_files}] {img_file.name} | {elapsed:.3f}s\033[K", end="", flush=True)
            
        except Exception as e:
            print(f"\n[!] Error on {img_file.name}: {e}")

    print("\nDone.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("-c", "--colors", type=int, default=4)
    parser.add_argument("-s", "--scale", type=int, default=4)
    parser.add_argument("--blend", type=float, default=0.05)
    
    # New Architecture Parameters
    parser.add_argument("--edge-weight", type=float, default=15.0, 
                        help="Anti-Gravity multiplier for high contrast pixels (default: 15.0)")
    parser.add_argument("--samples", type=int, default=5000, 
                        help="Number of pixels to sample for realtime simulation (default: 5000)")
    
    args = parser.parse_args()
    process_video_frames(args.input_dir, args.output_dir, args.colors, args.scale, args.blend, args.edge_weight, args.samples)