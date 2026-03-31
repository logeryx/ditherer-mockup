import argparse
import time
import numpy as np
from PIL import Image
from pathlib import Path
from scipy.cluster.vq import kmeans

def extract_stable_palette(img, num_colors=4, bin_size=32):
    """
    Uses K-Means clustering on BINNED unique colors. 
    This prevents massive gradients of similar colors (like dark walls) 
    from overwhelming the algorithm by collapsing them into a few single points.
    """
    small_img = img.resize((200, 200), resample=Image.Resampling.NEAREST)
    pixels = np.array(small_img).reshape(-1, 3)
    
    # --- THE NEW MAGIC: Color Binning / Posterization ---
    # We integer-divide by the bin_size, then multiply back.
    # E.g., with bin_size=32: values 0-31 all become 16, values 32-63 all become 48.
    # This aggressively averages thousands of similar dark shades into single colors.
    binned_pixels = (pixels // bin_size) * bin_size + (bin_size // 2)
    
    # Now extract the unique colors from the binned pixels.
    # A massive gradient wall will now only produce 2 or 3 unique points!
    unique_colors = np.unique(binned_pixels, axis=0).astype(float)
    
    # Fallback in the rare case the image has very few colors
    if len(unique_colors) < num_colors:
        unique_colors = pixels.astype(float)
        
    np.random.seed(42)
    
    # Run K-Means on this dramatically simplified color space
    palette_rgb, _ = kmeans(unique_colors, num_colors)
    
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
    
    # 1. PIXELATION: Downscale using BOX to perfectly average 4x4 blocks
    img_small = img.resize((small_width, small_height), resample=Image.Resampling.BOX)
    
    # 2. PREPARE PALETTE FOR PILLOW
    flat_palette = palette_rgb.flatten().tolist()
    # Pillow requires exactly 256 colors (768 values), pad the rest with black
    flat_palette += [0] * (768 - len(flat_palette))
    
    palette_img = Image.new('P', (1, 1))
    palette_img.putpalette(flat_palette)
    
    # 3. DITHER
    dithered_small = img_small.quantize(palette=palette_img, dither=Image.Dither.FLOYDSTEINBERG)
    
    # 4. UPSCALE: Use NEAREST to keep the pixels sharp and blocky
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
                
                # Pass the bin_size to our improved extraction function
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
    
    # Expose bin size as a command line argument so you can tune it
    parser.add_argument("-b", "--bin-size", type=int, default=32, 
                        help="Color grouping strength. Higher = more colors grouped (default: 32)")
    
    args = parser.parse_args()
    process_directory(args.input_dir, args.output_dir, args.colors, args.bin_size)