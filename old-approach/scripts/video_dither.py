import argparse
import time
import numpy as np
from PIL import Image
from pathlib import Path
from scipy.cluster.vq import kmeans

def sort_by_luminance(palette):
    """Sorts a palette from darkest to lightest based on human perception."""
    luminance = 0.299 * palette[:, 0] + 0.587 * palette[:, 1] + 0.114 * palette[:, 2]
    sorted_indices = np.argsort(luminance)
    return palette[sorted_indices]

def extract_stable_palette(img, num_colors=4, bin_size=32):
    """Extracts a palette by neutralizing population density."""
    small_img = img.resize((200, 200), resample=Image.Resampling.NEAREST)
    pixels = np.array(small_img).reshape(-1, 3)
    
    binned_pixels = (pixels // bin_size) * bin_size
    unique_bins, inverse_indices = np.unique(binned_pixels, axis=0, return_inverse=True)
    
    true_unique_colors = np.zeros_like(unique_bins, dtype=float)
    for i in range(len(unique_bins)):
        pixels_in_bin = pixels[inverse_indices == i]
        true_unique_colors[i] = pixels_in_bin.mean(axis=0)
    
    if len(true_unique_colors) < num_colors:
        true_unique_colors = pixels.astype(float)
        
    np.random.seed(42)
    palette_rgb, _ = kmeans(true_unique_colors, num_colors)
    
    return sort_by_luminance(palette_rgb)

def dither_pixelated(img, palette_rgb, scale_factor):
    """Downscales, applies Floyd-Steinberg dithering, and upscales cleanly."""
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
    dithered_final = dithered_small.resize((original_width, original_height), resample=Image.Resampling.NEAREST)
    
    return dithered_final

def process_directory(input_dir, output_dir, num_colors, bin_size, scale_factor, blend_factor):
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    
    out_path.mkdir(parents=True, exist_ok=True)
    valid_exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    
    files_to_process = sorted([f for f in in_path.iterdir() if f.suffix.lower() in valid_exts])
    total_files = len(files_to_process)
    
    if total_files == 0:
        print(f"No valid images found in {input_dir}")
        return

    # Ensure blend_factor is sensible (clamp between 0.01 and 1.0)
    blend_factor = max(0.01, min(1.0, blend_factor))

    previous_palette = None
    print(f"Starting batch process for {total_files} images...")
    print(f"Settings: Colors={num_colors}, BinSize={bin_size}, Scale={scale_factor}x, Blend={blend_factor*100}%")
    
    for i, img_file in enumerate(files_to_process, 1):
        t0 = time.time()
        
        try:
            img = Image.open(img_file).convert('RGB')
            current_palette = extract_stable_palette(img, num_colors=num_colors, bin_size=bin_size)
            
            # --- TEMPORAL EXPONENTIAL DECAY ---
            if previous_palette is not None:
                # Blend the new frame's palette into the historical palette
                current_palette = ((1.0 - blend_factor) * previous_palette) + (blend_factor * current_palette)
            
            previous_palette = current_palette
            final_palette_uint8 = np.clip(current_palette, 0, 255).astype(np.uint8)
            
            hex_colors = ['#' + ''.join(f'{c:02x}' for c in rgb) for rgb in final_palette_uint8]
            final_img = dither_pixelated(img, final_palette_uint8, scale_factor)
            
            out_file = out_path / f"{img_file.stem}-dither.png"
            final_img.save(out_file)
            
            elapsed = time.time() - t0
            
            status_line = f"\r[{i}/{total_files}] {img_file.name} | Colors: {' '.join(hex_colors)} | {elapsed:.2f}s"
            print(f"{status_line}\033[K", end="", flush=True)
            
        except Exception as e:
            print(f"\n[!] Error processing {img_file.name}: {e}")

    print(f"\nDone! Processed {total_files} files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video-Stable Palette Extraction & Pixelated Dithering")
    parser.add_argument("input_dir", help="Directory containing original images/frames")
    parser.add_argument("output_dir", help="Directory to save dithered images")
    parser.add_argument("-c", "--colors", type=int, default=4, help="Number of colors to extract (default: 4)")
    parser.add_argument("-b", "--bin-size", type=int, default=32, help="Color grouping strength (default: 32)")
    parser.add_argument("-s", "--scale", type=int, default=4, help="Pixelation scale factor. (default: 4)")
    
    # New argument for the Exponential Moving Average speed
    parser.add_argument("--blend", type=float, default=0.5, 
                        help="Palette reaction speed (0.01 to 1.0). Lower is slower/smoother. (default: 0.5)")
    
    args = parser.parse_args()
    process_directory(args.input_dir, args.output_dir, args.colors, args.bin_size, args.scale, args.blend)