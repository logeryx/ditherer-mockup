import argparse
import time
import numpy as np
from PIL import Image, ImageFilter
from pathlib import Path
from scipy.cluster.vq import kmeans2

# =====================================================================
# CIELAB COLOR SPACE CONVERSIONS
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
# NON-LINEAR LUMINANCE WARPING (The Brightness Trick)
# =====================================================================
def warp_luminance(l_array, bias):
    """Quadratic stretch: Bright colors are stretched far apart, darks stay together."""
    if bias <= 0: return l_array
    return l_array + (bias / 100.0) * (l_array ** 2)

def unwarp_luminance(l_warped_array, bias):
    """Exact mathematical inverse of the quadratic stretch."""
    if bias <= 0: return l_warped_array
    # Ensure no negative values slipped through due to K-Means averaging
    l_warped_array = np.maximum(0, l_warped_array)
    return (50.0 / bias) * (np.sqrt(1.0 + (4.0 * bias * l_warped_array) / 100.0) - 1.0)

# =====================================================================
# SALIENCY & EXTREME SAMPLING
# =====================================================================
def get_biased_samples(img, sample_size=5000, edge_weight=10.0):
    analysis_img = img.resize((128, 128), resample=Image.Resampling.BILINEAR)
    edges = analysis_img.convert('L').filter(ImageFilter.FIND_EDGES)
    l_channel = np.array(analysis_img.convert('L')).astype(float)
    
    pixels = np.array(analysis_img).reshape(-1, 3)
    edge_vals = np.array(edges).reshape(-1)
    l_vals = l_channel.reshape(-1)
    
    weights = np.ones(len(pixels))
    weights += (edge_vals / 255.0) * edge_weight
    
    # Boost extremes (darks and brights) slightly above edges
    extremes = np.abs(l_vals - 128.0) / 128.0 
    weights += extremes * (edge_weight * 1.5)
    
    probs = weights / np.sum(weights)
    actual_sample_size = min(sample_size, len(pixels))
    indices = np.random.choice(len(pixels), size=actual_sample_size, p=probs, replace=False)
    
    return pixels[indices]

# =====================================================================
# CORE QUANTIZATION PIPELINE WITH BRIGHT BIAS
# =====================================================================
def extract_palette(img, num_colors, prev_palette_lab=None, edge_weight=10.0, samples=5000, bright_bias=2.0):
    sampled_rgb = get_biased_samples(img, samples, edge_weight)
    sampled_lab = rgb_to_lab(sampled_rgb)
    
    # --- NON-LINEAR AXIS WARP ---
    sampled_lab[:, 0] = warp_luminance(sampled_lab[:, 0], bright_bias)
    
    if prev_palette_lab is not None and len(prev_palette_lab) == num_colors:
        seed_lab = prev_palette_lab.copy()
        seed_lab[:, 0] = warp_luminance(seed_lab[:, 0], bright_bias)
        centroids, _ = kmeans2(sampled_lab, seed_lab, iter=3, minit='matrix')
    else:
        try:
            centroids, _ = kmeans2(sampled_lab, num_colors, iter=15, minit='++')
        except ValueError:
            centroids, _ = kmeans2(sampled_lab, num_colors, iter=15, minit='points')
            
    # --- NON-LINEAR AXIS UN-WARP ---
    centroids[:, 0] = unwarp_luminance(centroids[:, 0], bright_bias)
    
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
def process_video_frames(input_dir, output_dir, num_colors, scale_factor, blend_factor, edge_weight, samples, bright_bias):
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    files_to_process = sorted([f for f in in_path.iterdir() if f.suffix.lower() in {'.png', '.jpg', '.jpeg'}])
    total_files = len(files_to_process)
    if total_files == 0: return

    prev_palette_rgb = None
    prev_palette_lab = None

    print(f"Starting Brightness-Biased CIELAB Pipeline on {total_files} frames...")

    for i, img_file in enumerate(files_to_process, 1):
        t0 = time.time()
        try:
            img = Image.open(img_file).convert('RGB')
            
            # Extract target with Bright Bias
            target_palette_lab = extract_palette(img, num_colors, prev_palette_lab, edge_weight, samples, bright_bias)
            target_palette_rgb = sort_by_luminance(lab_to_rgb(target_palette_lab))
            
            # EMA Temporal Blend
            if prev_palette_rgb is not None:
                final_palette_rgb = ((1.0 - blend_factor) * prev_palette_rgb) + (blend_factor * target_palette_rgb)
            else:
                final_palette_rgb = target_palette_rgb
                
            prev_palette_rgb = final_palette_rgb
            prev_palette_lab = rgb_to_lab(final_palette_rgb) 
            
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
    
    parser.add_argument("--edge-weight", type=float, default=15.0)
    parser.add_argument("--samples", type=int, default=5000)
    
    # New Adaptive Brightness Bias Parameter
    parser.add_argument("--bright-bias", type=float, default=2.0,
                        help="Quadratic multiplier to prioritize bright colors over shadows (default: 2.0)")
    
    args = parser.parse_args()
    process_video_frames(args.input_dir, args.output_dir, args.colors, args.scale, args.blend, args.edge_weight, args.samples, args.bright_bias)