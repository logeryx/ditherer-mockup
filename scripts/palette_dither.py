import numpy as np
from PIL import Image
import argparse
import time

def hex_to_rgb(hex_color):
    """Convert a hex color string to an RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def dither_pillow(image_path, palette_hex):
    """Fast built-in Floyd-Steinberg dithering using Pillow, with pixelation."""
    img = Image.open(image_path).convert('RGB')
    
    # --- 1. PIXELATION (Downscale) ---
    original_width, original_height = img.size
    
    # Calculate quater resolution
    small_width = original_width // 4
    small_height = original_height // 4
    
    # We use BOX resampling. When downsizing exactly by half, BOX takes the 
    # exact average of the 16 pixels (4x4 block), which is mathematically 
    # exactly what you asked for!
    img_small = img.resize((small_width, small_height), resample=Image.Resampling.BOX)
    
    # --- 2. DITHERING ---
    flat_palette = []
    for h in palette_hex:
        flat_palette.extend(hex_to_rgb(h))
        
    # Pillow requires exactly 256 colors (768 values)
    flat_palette += [0] * (768 - len(flat_palette))
    
    palette_img = Image.new('P', (1, 1))
    palette_img.putpalette(flat_palette)
    
    # Dither the smaller, low-res image
    dithered_small = img_small.quantize(palette=palette_img, dither=Image.Dither.FLOYDSTEINBERG)
    
    # --- 3. UPSCALE TO ORIGINAL SIZE ---
    # We MUST use NEAREST resampling here. This ensures the pixels are just 
    # stretched into sharp 4x4 blocks. Any other resampling would blur the dither!
    dithered_final = dithered_small.resize((original_width, original_height), resample=Image.Resampling.NEAREST)
    
    return dithered_final

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dither an image using a custom palette with 2x pixelation.")
    parser.add_argument("input", help="Path to the input image file")
    parser.add_argument("output", help="Path to save the output image file")
    parser.add_argument("--palette", nargs='+', required=True, 
                        help="List of hex colors, e.g., --palette #000000 #FFFFFF #FF0000")
    
    args = parser.parse_args()

    print(f"  -> Dithering '{args.input}' with palette {args.palette}")
    t0 = time.time()
    
    try:
        result = dither_pillow(args.input, args.palette)
        result.save(args.output)
        print(f"  -> Saved to '{args.output}' (took {time.time()-t0:.3f}s)")
    except Exception as e:
        print(f"  -> Error processing image: {e}")