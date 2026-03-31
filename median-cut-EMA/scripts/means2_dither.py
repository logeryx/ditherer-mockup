import argparse
import time
import numpy as np
from PIL import Image
from pathlib import Path
from scipy.cluster.vq import kmeans

def extract_stable_palette(img, num_colors=4):
    """
    Uses K-Means clustering to find highly stable representative colors.
    """
    # 1. Shrink image drastically for fast color clustering.
    # We don't need full resolution to find the dominant colors.
    small_img = img.resize((100, 100), resample=Image.Resampling.BILINEAR)
    
    # 2. Convert to numpy array of flat RGB pixels
    pixels = np.array(small_img, dtype=float).reshape(-1, 3)
    
    # 3. Lock the random seed. This guarantees that if you feed it the exact 
    # same image twice, it initializes the exact same way.
    np.random.seed(42)
    
    # 4. Run K-Means clustering (returns the palette and the distortion/error)
    palette_rgb, _ = kmeans(pixels, num_colors)
    
    # Convert float arrays back to standard 0-255 RGB integers
    palette_rgb = np.clip(palette_rgb, 0, 255).astype(np.uint8)
    return palette_rgb

def dither_pixelated(img, palette_rgb):
    """
    Downscales by 4x, applies Floyd-Steinberg dithering using the palette, 
    and upscales cleanly back to original size.
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

def process_directory(input_dir, output_dir, num_colors):
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    
    out_path.mkdir(parents=True, exist_ok=True)
    valid_exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    
    for img_file in in_path.iterdir():
        if img_file.suffix.lower() in valid_exts:
            print(f"\n--- Processing {img_file.name} ---")
            t0 = time.time()
            
            try:
                # Load image once
                img = Image.open(img_file).convert('RGB')
                
                # Extract the stable palette
                palette_rgb = extract_stable_palette(img, num_colors=num_colors)
                
                # Print hex codes just so you can see what it chose
                hex_colors = ['#' + ''.join(f'{c:02x}' for c in rgb) for rgb in palette_rgb]
                print(f"  -> Generated Palette: {' '.join(hex_colors)}")
                
                # Dither and pixelate
                final_img = dither_pixelated(img, palette_rgb)
                
                # Save
                out_file = out_path / f"{img_file.stem}-dither.png"
                final_img.save(out_file)
                
                print(f"  -> Saved to '{out_file.name}' (took {time.time()-t0:.3f}s)")
                
            except Exception as e:
                print(f"  [!] Error processing {img_file.name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Palette Extraction & Pixelated Dithering")
    parser.add_argument("input_dir", help="Directory containing original images")
    parser.add_argument("output_dir", help="Directory to save dithered images")
    parser.add_argument("-c", "--colors", type=int, default=4, help="Number of colors to extract (default: 4)")
    
    args = parser.parse_args()
    process_directory(args.input_dir, args.output_dir, args.colors)