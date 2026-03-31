import argparse
import time
import numpy as np
from PIL import Image
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
# DENSITY NEUTRALIZATION (The "Look" you wanted)
# =====================================================================
def get_unique_binned_colors(img, bin_size=32):
    """
    Destroys population gravity entirely. 50,000 dark pixels become 1 vote.
    50 torch pixels spanning yellow and orange become 2 votes.
    """
    small_img = img.resize((200, 200), resample=Image.Resampling.NEAREST)
    pixels = np.array(small_img).reshape(-1, 3)
    
    # Group similar colors into rigid bins
    binned_pixels = (pixels // bin_size) * bin_size
    
    # Extract only the unique bins (Population gravity is now dead)
    unique_bins, inverse_indices = np.unique(binned_pixels, axis=0, return_inverse=True)
    
    # Calculate the true average RGB of the pixels that fell into each bin
    # This keeps the colors looking accurate rather than artificially snapped to a grid
    true_colors = np.zeros_like(unique_bins, dtype=float)
    for i in range(len(unique_bins)):
        true_colors[i] = pixels[inverse_indices == i].mean(axis=0)
        
    return true_colors

# =====================================================================
# CORE QUANTIZATION PIPELINE
# =====================================================================
def extract_palette(img, num_colors, prev_palette_lab=None, bin_size=32):
    # 1. Get our gravity-free true colors
    true_colors_rgb = get_unique_binned_colors(img, bin_size)
    
    # Fallback if image is literally a solid color
    if len(true_colors_rgb) < num_colors:
        return rgb_to_lab(np.tile(true_colors_rgb[0], (num_colors, 1)))
        
    # 2. Convert to LAB space for accurate human-perception clustering
    true_colors_lab = rgb_to_lab(true_colors_rgb)
    
    # 3. CLUSTERING WITH STABILITY SEEDING
    # Because all points now have a weight of exactly 1, K-Means will flawlessly
    # divide the slots among the distinct color concepts (Torch, Shadow, Midtone).
    if prev_palette_lab is not None and len(prev_palette_lab) == num_colors:
        # Pass previous frame's palette to guarantee zero flashing!
        centroids, _ = kmeans2(true_colors_lab, prev_palette_lab, iter=5, minit='matrix')
    else:
        try:
            centroids, _ = kmeans2(true_colors_lab, num_colors, iter=20, minit='++')
        except ValueError:
            centroids, _ = kmeans2(true_colors_lab, num_colors, iter=20, minit='points')
            
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
def process_video_frames(input_dir, output_dir, num_colors, scale_factor, blend_factor, bin_size):
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    files_to_process = sorted([f for f in in_path.iterdir() if f.suffix.lower() in {'.png', '.jpg', '.jpeg'}])
    total_files = len(files_to_process)
    if total_files == 0: return

    prev_palette_rgb = None
    prev_palette_lab = None

    print(f"Starting Stable Binned CIELAB Pipeline on {total_files} frames...")

    for i, img_file in enumerate(files_to_process, 1):
        t0 = time.time()
        try:
            img = Image.open(img_file).convert('RGB')
            
            # 1. Extract
            target_palette_lab = extract_palette(img, num_colors, prev_palette_lab, bin_size)
            target_palette_rgb = sort_by_luminance(lab_to_rgb(target_palette_lab))
            
            # 2. Smooth EMA Temporal Blend
            if prev_palette_rgb is not None:
                final_palette_rgb = ((1.0 - blend_factor) * prev_palette_rgb) + (blend_factor * target_palette_rgb)
            else:
                final_palette_rgb = target_palette_rgb
                
            # Update States
            prev_palette_rgb = final_palette_rgb
            prev_palette_lab = rgb_to_lab(final_palette_rgb) 
            
            # 3. Dither
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
    
    # Bringing back the magic parameter!
    parser.add_argument("-b", "--bin-size", type=int, default=32,
                        help="Grouping strength. Higher = shadows merge into 1 color, torch gets 2 colors (default: 32)")
    
    args = parser.parse_args()
    process_video_frames(args.input_dir, args.output_dir, args.colors, args.scale, args.blend, args.bin_size)