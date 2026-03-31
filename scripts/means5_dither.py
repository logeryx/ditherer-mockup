import argparse
import time
import numpy as np
from PIL import Image
from pathlib import Path
from scipy.cluster.vq import kmeans

def extract_stable_palette(img, num_colors=4, bin_size=32):
    """
    Extracts a palette by neutralizing population density (massive walls) 
    while preserving the EXACT true colors of the original image.
    """
    small_img = img.resize((200, 200), resample=Image.Resampling.NEAREST)
    pixels = np.array(small_img).reshape(-1, 3)
    
    # 1. Map pixels to their bins just to see who belongs together
    binned_pixels = (pixels // bin_size) * bin_size
    
    # 2. Find the unique bins, and get the inverse indices so we know 
    # which original pixel belongs to which bin
    unique_bins, inverse_indices = np.unique(binned_pixels, axis=0, return_inverse=True)
    
    # 3. THE TRUE COLOR MAGIC:
    # Instead of using the artificial bin colors, we calculate the exact 
    # average of the ORIGINAL pixels that fell into each bin.
    true_unique_colors = np.zeros_like(unique_bins, dtype=float)
    for i in range(len(unique_bins)):
        # Grab all original pixels in this bin and find their true average color
        pixels_in_bin = pixels[inverse_indices == i]
        true_unique_colors[i] = pixels_in_bin.mean(axis=0)
    
    # Fallback in the rare case the image has very few color concepts
    if len(true_unique_colors) < num_colors:
        true_unique_colors = pixels.astype(float)
        
    np.random.seed(42)
    
    # 4. Run K-Means on the true unique colors
    # Now it is clustering actual colors from your image, completely free 
    # from the 512-color grid, and completely free from population gravity!
    palette_rgb, _ = kmeans(true_unique_colors, num_colors)
    
    palette_rgb = np.clip(palette_rgb, 0, 255).astype(np.uint8)
    return palette_rgb

def dither_pixelated(img, palette_rgb):
    """
    Downscales by 4x (16 pixels to 1), applies Floyd-Steinberg dithering 
    using the palette, and upscales cleanly back to original size.
    """
    original_width, original_height = img.size
    small_width = original_width // 4
    small_height = original_height // 4
    
    img_small = img.resize((small_width, small_height), resample=Image.Resampling.BOX)
    
    flat_palette = palette_rgb.flatten().tolist()
    flat_palette += [0] * (768 - len(flat_palette))
    
    palette_img = Image.new('P', (1, 1))
    palette_img.putpalette(flat_palette)
    
    dithered_small = img_small.quantize(palette=palette_img, dither=Image.Dither.FLOYDSTEINBERG)
    
    dithered_final = dithered_small.resize((original_width, original_height), resample=Image.Resampling.NEAREST)
    
    return dithered_final

def process_directory(input_dir, output_dir, num_colors, bin_size):
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    
    out_path.mkdir(parents=True, exist_ok=True)
    valid_exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    
    for img_file in in_path.iterdir():
        if img_file.suffix.lower() in valid_exts:
            print(f"\n--- Processing {img_file.name} ---")
            t0 = time.time()
            
            try:
                img = Image.open(img_file).convert('RGB')
                
                palette_rgb = extract_stable_palette(img, num_colors=num_colors, bin_size=bin_size)
                
                hex_colors = ['#' + ''.join(f'{c:02x}' for c in rgb) for rgb in palette_rgb]
                print(f"  -> Generated Palette: {' '.join(hex_colors)}")
                
                final_img = dither_pixelated(img, palette_rgb)
                
                out_file = out_path / f"{img_file.stem}_0dither.png"
                final_img.save(out_file)
                
                print(f"  -> Saved to '{out_file.name}' (took {time.time()-t0:.3f}s)")
                
            except Exception as e:
                print(f"  [!] Error processing {img_file.name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Palette Extraction & Pixelated Dithering")
    parser.add_argument("input_dir", help="Directory containing original images")
    parser.add_argument("output_dir", help="Directory to save dithered images")
    parser.add_argument("-c", "--colors", type=int, default=4, help="Number of colors to extract (default: 4)")
    parser.add_argument("-b", "--bin-size", type=int, default=32, 
                        help="Color grouping strength. Higher = more colors grouped (default: 32)")
    
    args = parser.parse_args()
    process_directory(args.input_dir, args.output_dir, args.colors, args.bin_size)