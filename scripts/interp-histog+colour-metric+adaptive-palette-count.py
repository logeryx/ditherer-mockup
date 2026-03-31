import argparse
import time
import numpy as np
from PIL import Image
from pathlib import Path

# =====================================================================
# COLOR SPACE MATH & OPTIMIZATION
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

def rgb_to_lab(rgb_array):
    """Vectorized conversion from RGB (0-255) to CIE-L*a*b* space."""
    rgb = rgb_array.astype(float) / 255.0
    
    # RGB to XYZ
    mask = rgb > 0.04045
    rgb[mask] = ((rgb[mask] + 0.055) / 1.055) ** 2.4
    rgb[~mask] = rgb[~mask] / 12.92
    rgb *= 100.0

    xyz = np.zeros_like(rgb)
    xyz[:, 0] = rgb[:, 0] * 0.4124564 + rgb[:, 1] * 0.3575761 + rgb[:, 2] * 0.1804375
    xyz[:, 1] = rgb[:, 0] * 0.2126729 + rgb[:, 1] * 0.7151522 + rgb[:, 2] * 0.0721750
    xyz[:, 2] = rgb[:, 0] * 0.0193339 + rgb[:, 1] * 0.1191920 + rgb[:, 2] * 0.9503041

    # XYZ to LAB
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

# =====================================================================
# HISTOGRAM ECONOMY
# =====================================================================
def get_truth_histogram(img, bin_size=32):
    small_img = img.resize((200, 200), resample=Image.Resampling.NEAREST)
    pixels = np.array(small_img).reshape(-1, 3)
    binned_pixels = (pixels // bin_size) * bin_size
    unique_colors, counts = np.unique(pack_rgb(binned_pixels), return_counts=True)
    return unique_colors, counts

def evolve_histogram(prev_colors, prev_counts, truth_colors, truth_counts, transfer_rate=0.05):
    all_colors = np.unique(np.concatenate([prev_colors, truth_colors]))
    
    P_counts_aligned = np.zeros(len(all_colors), dtype=int)
    P_counts_aligned[np.searchsorted(all_colors, prev_colors)] = prev_counts
    
    T_counts_aligned = np.zeros(len(all_colors), dtype=int)
    T_counts_aligned[np.searchsorted(all_colors, truth_colors)] = truth_counts
    
    deltas = T_counts_aligned - P_counts_aligned
    over_idx = np.where(deltas < 0)[0]
    under_idx = np.where(deltas > 0)[0]
    
    over_sorted = over_idx[np.argsort(P_counts_aligned[over_idx])]
    under_sorted = under_idx[np.argsort(-deltas[under_idx])]
    
    max_transfer = int(np.sum(prev_counts) * transfer_rate)
    
    freed_volume = 0
    for idx in over_sorted:
        take = min(-deltas[idx], max_transfer - freed_volume)
        P_counts_aligned[idx] -= take
        freed_volume += take
        if freed_volume >= max_transfer: break
            
    volume_to_give = freed_volume
    for idx in under_sorted:
        give = min(deltas[idx], volume_to_give)
        P_counts_aligned[idx] += give
        volume_to_give -= give
        if volume_to_give <= 0: break
            
    valid_mask = P_counts_aligned > 0
    return all_colors[valid_mask], P_counts_aligned[valid_mask]

# =====================================================================
# ADAPTIVE TARGET QUANTIZER
# =====================================================================
def generate_adaptive_palette(colors_packed, counts, target_colors, max_dist):
    """
    Splits color boxes until we hit the target color count. 
    If the maximum L*a*b* error still exceeds max_dist, it CHEATS 
    and adds more colors until the distance constraint is met.
    """
    colors_rgb = unpack_rgb(colors_packed).astype(float)
    colors_lab = rgb_to_lab(colors_rgb)
    
    # L_WEIGHT acts as a multiplier. By weighting Luminance higher (e.g. 1.5), 
    # the algorithm treats slight brightness changes as more drastic than color changes.
    L_WEIGHT = 1.5 
    
    # Box definition: (rgb_array, lab_array, counts)
    boxes = [(colors_rgb, colors_lab, counts)]
    
    while True:
        box_errors = []
        for c_rgb, c_lab, cnt in boxes:
            if len(c_rgb) <= 1:
                box_errors.append(0.0)
                continue
                
            avg_lab = np.sum(c_lab * cnt[:, np.newaxis], axis=0) / np.sum(cnt)
            
            # Calculate error with weighted Luminance
            diff = (c_lab - avg_lab) * np.array([L_WEIGHT, 1.0, 1.0])
            dists = np.linalg.norm(diff, axis=1)
            box_errors.append(np.max(dists))
            
        max_error_idx = np.argmax(box_errors)
        max_error = box_errors[max_error_idx]
        
        # STOPPING CONDITIONS
        # We hit our target color count AND we satisfy the distance metric
        if len(boxes) >= target_colors and max_error <= max_dist:
            break
        if len(boxes) >= 64: # Absolute safety limit to prevent explosion
            break
        if max_error == 0:   # Perfect quantization achieved
            break
            
        # POP box with worst error
        c_rgb, c_lab, cnt = boxes.pop(max_error_idx)
        
        # Choose axis to split based on L*a*b* variance (with L weighted!)
        l_range = (c_lab[:,0].max() - c_lab[:,0].min()) * L_WEIGHT
        a_range = c_lab[:,1].max() - c_lab[:,1].min()
        b_range = c_lab[:,2].max() - c_lab[:,2].min()
        
        max_range = max(l_range, a_range, b_range)
        split_channel = 0 if max_range == l_range else (1 if max_range == a_range else 2)
        
        # Sort colors by chosen L*a*b* channel
        sort_idx = np.argsort(c_lab[:, split_channel])
        c_rgb_sorted, c_lab_sorted, cnt_sorted = c_rgb[sort_idx], c_lab[sort_idx], cnt[sort_idx]
        channel_vals = c_lab_sorted[:, split_channel]
        
        # --- GAP FINDING MAGIC ---
        cum_counts = np.cumsum(cnt_sorted)
        total_pop = cum_counts[-1]
        
        # Search window: Look between the 25% and 75% population marks
        q25_idx = np.searchsorted(cum_counts, total_pop * 0.25)
        q75_idx = np.searchsorted(cum_counts, total_pop * 0.75)
        
        split_idx = np.searchsorted(cum_counts, total_pop * 0.5) # Default: Median
        
        if q75_idx > q25_idx + 1:
            window_vals = channel_vals[q25_idx:q75_idx+1]
            gaps = np.diff(window_vals)
            max_gap_idx = np.argmax(gaps)
            
            # If a distinct gap exists in the middle mass, split the gap instead of splitting the median mass
            if gaps[max_gap_idx] > 0.5: 
                split_idx = q25_idx + max_gap_idx + 1

        # Fallbacks
        split_idx = max(1, min(split_idx, len(c_rgb_sorted) - 1))
        
        boxes.append((c_rgb_sorted[:split_idx], c_lab_sorted[:split_idx], cnt_sorted[:split_idx]))
        boxes.append((c_rgb_sorted[split_idx:], c_lab_sorted[split_idx:], cnt_sorted[split_idx:]))
        
    # Generate final palette
    palette = []
    for c_rgb, _, cnt in boxes:
        avg_color = np.sum(c_rgb * cnt[:, np.newaxis], axis=0) / np.sum(cnt)
        palette.append(avg_color)
        
    palette_rgb = np.array(palette, dtype=np.uint8)
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
    # Safely pad to Pillow's rigid 256 color requirement
    if len(flat_palette) < 768:
        flat_palette += [0] * (768 - len(flat_palette))
    else:
        flat_palette = flat_palette[:768]
    
    palette_img = Image.new('P', (1, 1))
    palette_img.putpalette(flat_palette)
    
    dithered_small = img_small.quantize(palette=palette_img, dither=Image.Dither.FLOYDSTEINBERG)
    return dithered_small.resize((original_width, original_height), resample=Image.Resampling.NEAREST)


def process_video_frames(input_dir, output_dir, num_colors, max_dist, bin_size, scale_factor, blend_factor, transfer_rate):
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

    print(f"Starting Adaptive Histogram Pipeline on {total_files} frames...")

    for i, img_file in enumerate(files_to_process, 1):
        t0 = time.time()
        
        try:
            img = Image.open(img_file).convert('RGB')
            
            new_truth_colors, new_truth_counts = get_truth_histogram(img, bin_size)
            
            if prev_interp_colors is None:
                new_interp_colors, new_interp_counts = new_truth_colors, new_truth_counts
            else:
                new_interp_colors, new_interp_counts = evolve_histogram(
                    prev_interp_colors, prev_interp_counts, new_truth_colors, new_truth_counts, transfer_rate
                )
            
            target_palette_rgb = generate_adaptive_palette(
                new_interp_colors, new_interp_counts, num_colors, max_dist
            )
            
            # --- DYNAMIC LENGTH EMA ---
            if prev_palette_rgb is not None:
                final_palette_rgb = np.zeros_like(target_palette_rgb, dtype=float)
                # If a torch appears, the palette might grow from 4 to 5 colors.
                # For each target color, we find the closest "ancestor" color in the previous frame
                # to blend from, ensuring smooth fade-ins for entirely new colors!
                for j, target_c in enumerate(target_palette_rgb):
                    dists = np.linalg.norm(prev_palette_rgb - target_c, axis=1)
                    closest_ancestor_idx = np.argmin(dists)
                    final_palette_rgb[j] = ((1.0 - blend_factor) * prev_palette_rgb[closest_ancestor_idx]) + (blend_factor * target_c)
            else:
                final_palette_rgb = target_palette_rgb.astype(float)
                
            prev_interp_colors, prev_interp_counts = new_interp_colors, new_interp_counts
            prev_palette_rgb = final_palette_rgb
            
            final_palette_uint8 = np.clip(final_palette_rgb, 0, 255).astype(np.uint8)
            final_img = dither_pixelated(img, final_palette_uint8, scale_factor)
            final_img.save(out_path / f"{img_file.stem}-dither.png")
            
            # Progress print
            current_count = len(final_palette_uint8)
            cheat_status = f"(CHEATED! Aim={num_colors})" if current_count > num_colors else f"(Target Hit)"
            print(f"\r[{i}/{total_files}] {img_file.name} | Using {current_count} colors {cheat_status} | {time.time()-t0:.2f}s\033[K", end="", flush=True)
            
        except Exception as e:
            print(f"\n[!] Error on {img_file.name}: {e}")

    print("\nDone.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("-c", "--colors", type=int, default=4, help="Target minimum colors")
    parser.add_argument("-d", "--max-dist", type=float, default=25.0, help="Max L*a*b* error before cheating and adding colors")
    parser.add_argument("-b", "--bin-size", type=int, default=32)
    parser.add_argument("-s", "--scale", type=int, default=4)
    parser.add_argument("--blend", type=float, default=0.05)
    parser.add_argument("--transfer", type=float, default=0.05)
    
    args = parser.parse_args()
    process_video_frames(args.input_dir, args.output_dir, args.colors, args.max_dist, args.bin_size, args.scale, args.blend, args.transfer)