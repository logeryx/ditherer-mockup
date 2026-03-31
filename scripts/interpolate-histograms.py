import argparse
import time
import numpy as np
from PIL import Image
from pathlib import Path

# =====================================================================
# FAST COLOR PACKING (NumPy Optimization)
# Packs [R, G, B] into a 32-bit integer. This turns a 3D histogram 
# into a 1D array, making sorting and matching millions of times faster.
# =====================================================================
def pack_rgb(rgb_array):
    return (rgb_array[:, 0].astype(np.uint32) << 16) | \
           (rgb_array[:, 1].astype(np.uint32) << 8) | \
            rgb_array[:, 2].astype(np.uint32)

def unpack_rgb(packed_array):
    return np.column_stack((
        (packed_array >> 16) & 255,
        (packed_array >> 8) & 255,
         packed_array & 255
    )).astype(np.uint8)

# =====================================================================
# HISTOGRAM PIPELINE
# =====================================================================
def get_truth_histogram(img, bin_size=32):
    """Calculates the raw truth histogram for a single frame."""
    small_img = img.resize((200, 200), resample=Image.Resampling.NEAREST)
    pixels = np.array(small_img).reshape(-1, 3)
    
    binned_pixels = (pixels // bin_size) * bin_size
    packed_pixels = pack_rgb(binned_pixels)
    
    unique_colors, counts = np.unique(packed_pixels, return_counts=True)
    return unique_colors, counts

def evolve_histogram(prev_colors, prev_counts, truth_colors, truth_counts, transfer_rate=0.05):
    """
    THE FLUID ECONOMY LOGIC: 
    Slowly spills pixel volume from over-represented old colors 
    into under-represented new colors, maintaining exactly constant volume.
    """
    # 1. Align the color spaces
    # Create a superset of all colors present in either memory or truth
    all_colors = np.unique(np.concatenate([prev_colors, truth_colors]))
    
    P_counts_aligned = np.zeros(len(all_colors), dtype=int)
    idx_prev = np.searchsorted(all_colors, prev_colors)
    P_counts_aligned[idx_prev] = prev_counts
    
    T_counts_aligned = np.zeros(len(all_colors), dtype=int)
    idx_truth = np.searchsorted(all_colors, truth_colors)
    T_counts_aligned[idx_truth] = truth_counts
    
    # Positive delta = We need MORE of this color (spill into this)
    # Negative delta = We have TOO MUCH of this color (drain from this)
    deltas = T_counts_aligned - P_counts_aligned
    
    # 2. Identify the groups
    over_idx = np.where(deltas < 0)[0]
    under_idx = np.where(deltas > 0)[0]
    
    # 3. Sort by priority
    # Drain the LEAST frequent over-represented colors first
    over_sorted = over_idx[np.argsort(P_counts_aligned[over_idx])]
    
    # Fill the MOST heavily missing under-represented colors first
    under_sorted = under_idx[np.argsort(-deltas[under_idx])]
    
    # 4. Calculate the maximum volume we are allowed to transfer this frame
    max_transfer = int(np.sum(prev_counts) * transfer_rate)
    
    # --- PHASE 1: DRAIN ---
    freed_volume = 0
    for idx in over_sorted:
        available_to_take = -deltas[idx]
        take = min(available_to_take, max_transfer - freed_volume)
        
        P_counts_aligned[idx] -= take
        freed_volume += take
        
        if freed_volume >= max_transfer:
            break
            
    # --- PHASE 2: FILL ---
    volume_to_give = freed_volume
    for idx in under_sorted:
        need = deltas[idx]
        give = min(need, volume_to_give)
        
        P_counts_aligned[idx] += give
        volume_to_give -= give
        
        if volume_to_give <= 0:
            break
            
    # Clean up any colors that were completely drained to 0
    valid_mask = P_counts_aligned > 0
    final_colors = all_colors[valid_mask]
    final_counts = P_counts_aligned[valid_mask]
    
    return final_colors, final_counts

# =====================================================================
# SIMPLEST MEDIAN CUT ALGORITHM
# =====================================================================
def generate_palette_median_cut(colors_packed, counts, num_colors):
    """
    Standard Median Cut relying heavily on pixel gravity (counts).
    Splits color space along the longest axis based on population medians.
    """
    colors_rgb = unpack_rgb(colors_packed).astype(float)
    
    # A box is defined by its colors and their respective counts
    boxes = [(colors_rgb, counts)]
    
    while len(boxes) < num_colors:
        largest_range = -1
        box_to_split_idx = -1
        split_channel = -1
        
        # Find the box with the widest color variance
        for i, (c, cnt) in enumerate(boxes):
            if len(c) <= 1: continue # Can't split a single color
            
            r_range = c[:,0].max() - c[:,0].min()
            g_range = c[:,1].max() - c[:,1].min()
            b_range = c[:,2].max() - c[:,2].min()
            
            max_r = max(r_range, g_range, b_range)
            
            if max_r > largest_range:
                largest_range = max_r
                box_to_split_idx = i
                # Note which axis is the longest (R=0, G=1, B=2)
                if max_r == r_range: split_channel = 0
                elif max_r == g_range: split_channel = 1
                else: split_channel = 2
                
        if box_to_split_idx == -1:
            break # No boxes can be split further
            
        c, cnt = boxes.pop(box_to_split_idx)
        
        # Sort the colors in this box by the longest channel
        sort_idx = np.argsort(c[:, split_channel])
        c_sorted = c[sort_idx]
        cnt_sorted = cnt[sort_idx]
        
        # Split at the POPULATION median (where the pixel count reaches 50%)
        cum_counts = np.cumsum(cnt_sorted)
        half_pop = cum_counts[-1] / 2.0
        split_idx = np.searchsorted(cum_counts, half_pop)
        
        # Ensure we don't create empty boxes
        if split_idx == 0: split_idx = 1
        if split_idx == len(c_sorted): split_idx = len(c_sorted) - 1
        
        boxes.append((c_sorted[:split_idx], cnt_sorted[:split_idx]))
        boxes.append((c_sorted[split_idx:], cnt_sorted[split_idx:]))
        
    # Generate the final palette by averaging each box
    palette = []
    for c, cnt in boxes:
        weighted_sum = np.sum(c * cnt[:, np.newaxis], axis=0)
        avg_color = weighted_sum / np.sum(cnt)
        palette.append(avg_color)
        
    palette_rgb = np.array(palette, dtype=np.uint8)
    
    # Sort by luminance for consistent temporal mapping if blended
    luminance = 0.299 * palette_rgb[:, 0] + 0.587 * palette_rgb[:, 1] + 0.114 * palette_rgb[:, 2]
    return palette_rgb[np.argsort(luminance)]

# =====================================================================
# DITHERING & BATCH LOGIC
# =====================================================================
def dither_pixelated(img, palette_rgb, scale_factor):
    original_width, original_height = img.size
    
    scale_factor = max(1, scale_factor)
    small_width = max(1, original_width // scale_factor)
    small_height = max(1, original_height // scale_factor)
    
    img_small = img.resize((small_width, small_height), resample=Image.Resampling.BOX)
    
    flat_palette = palette_rgb.flatten().tolist()
    flat_palette += [0] * (768 - len(flat_palette))
    
    palette_img = Image.new('P', (1, 1))
    palette_img.putpalette(flat_palette)
    
    dithered_small = img_small.quantize(palette=palette_img, dither=Image.Dither.FLOYDSTEINBERG)
    return dithered_small.resize((original_width, original_height), resample=Image.Resampling.NEAREST)


def process_video_frames(input_dir, output_dir, num_colors, bin_size, scale_factor, blend_factor, transfer_rate):
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    files_to_process = sorted([f for f in in_path.iterdir() if f.suffix.lower() in {'.png', '.jpg', '.jpeg'}])
    total_files = len(files_to_process)
    
    if total_files == 0:
        print("No valid images found.")
        return

    prev_interp_colors = None
    prev_interp_counts = None
    prev_palette_rgb = None

    print(f"Starting Fluid Histogram Pipeline on {total_files} frames...")

    for i, img_file in enumerate(files_to_process, 1):
        t0 = time.time()
        
        try:
            img = Image.open(img_file).convert('RGB')
            
            # 1. Get NEW TRUTH
            new_truth_colors, new_truth_counts = get_truth_histogram(img, bin_size)
            
            # 2. INTERPOLATE HISTOGRAM
            if prev_interp_colors is None:
                new_interp_colors = new_truth_colors
                new_interp_counts = new_truth_counts
            else:
                new_interp_colors, new_interp_counts = evolve_histogram(
                    prev_interp_colors, prev_interp_counts, 
                    new_truth_colors, new_truth_counts, 
                    transfer_rate=transfer_rate
                )
            
            # 3. GENERATE TARGET PALETTE using Simplest Median Cut
            target_palette_rgb = generate_palette_median_cut(
                new_interp_colors, new_interp_counts, num_colors
            )
            
            # 4. FINAL PALETTE INTERPOLATION (The EMA blend)
            if prev_palette_rgb is not None:
                final_palette_rgb = ((1.0 - blend_factor) * prev_palette_rgb) + (blend_factor * target_palette_rgb)
            else:
                final_palette_rgb = target_palette_rgb
                
            prev_interp_colors = new_interp_colors
            prev_interp_counts = new_interp_counts
            prev_palette_rgb = final_palette_rgb
            
            # 5. DITHER
            final_palette_uint8 = np.clip(final_palette_rgb, 0, 255).astype(np.uint8)
            final_img = dither_pixelated(img, final_palette_uint8, scale_factor)
            
            out_file = out_path / f"{img_file.stem}-dither.png"
            final_img.save(out_file)
            
            elapsed = time.time() - t0
            print(f"\r[{i}/{total_files}] {img_file.name} | Active Colors in Memory: {len(new_interp_colors)} | {elapsed:.2f}s\033[K", end="", flush=True)
            
        except Exception as e:
            print(f"\n[!] Error on {img_file.name}: {e}")

    print("\nDone.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("-c", "--colors", type=int, default=4)
    parser.add_argument("-b", "--bin-size", type=int, default=32)
    parser.add_argument("-s", "--scale", type=int, default=4)
    parser.add_argument("--blend", type=float, default=0.5)
    
    # The 5% Volume Transfer parameter
    parser.add_argument("--transfer", type=float, default=0.05, 
                        help="Ratio of total pixel volume allowed to be swapped per frame (default: 0.05)")
    
    args = parser.parse_args()
    process_video_frames(args.input_dir, args.output_dir, args.colors, args.bin_size, args.scale, args.blend, args.transfer)