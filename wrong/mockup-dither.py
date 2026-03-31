import numpy as np
from PIL import Image
from scipy.spatial import ckdtree
import time
import argparse

def hex_to_rgb(hex_color):
    """Convert a hex color string to an RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def dither_pillow(image_path, palette_hex):
    """
    Approach 1: The Fast Library Method
    Uses Pillow's built-in quantization and Floyd-Steinberg dithering.
    Extremely fast, but abstracts away the underlying math.
    """
    print("Running Pillow built-in dithering...")
    img = Image.open(image_path).convert('RGB')
    
    # Flatten the palette into a 1D list [R, G, B, R, G, B...]
    flat_palette = []
    for h in palette_hex:
        flat_palette.extend(hex_to_rgb(h))
        
    # Pillow requires the palette array to have exactly 256 colors (768 values)
    # We pad the rest with zeros (black)
    flat_palette += [0] * (768 - len(flat_palette))
    
    # Create a dummy image to hold the palette
    palette_img = Image.new('P', (1, 1))
    palette_img.putpalette(flat_palette)
    
    # Quantize the original image to our custom palette using Floyd-Steinberg dithering
    dithered = img.quantize(palette=palette_img, dither=Image.Dither.FLOYDSTEINBERG)
    return dithered

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dither an image using a custom palette.")
    parser.add_argument("input", help="Path to the input image file (e.g., input.png)")
    parser.add_argument("output", help="Path to save the output image file (e.g., output.png)")
    args = parser.parse_args()
    
    # Some example palettes to test with:
    PALETTE_CGA = ['#000000', '#55FFFF', '#FF55FF', '#FFFFFF'] # Standard 4-color CGA
    PALETTE_GAMEBOY = ['#0f380f', '#306230', '#8bac0f', '#9bbc0f'] # Nintendo Gameboy
    PALETTE_CUSTOM = ['#609DA6', '#458191', '#31566E', '#12171F'] # Custom song of the sea
    PALETTE_CUSTOM2 = ["#BBDDB4", "#5BA25D", "#75AC86", "#05181F"] # Custom  1151261
    PALETTE_CUSTOM3 = ["#839D75", "#507C42", "#508D68", "#0C2F28"] # Custom  1151263
    PALETTE_CUSTOM4 = ["#231E0B", "#6D6E3F", "#4B541E", "#AEB78C"] # Custom  83...
    PALETTE_CUSTOM5 = ["#9D734B", "#6E6653", "#45534D", "#080914"] # Custom  wall...
    
    CHOSEN_PALETTE = PALETTE_CUSTOM5

    print(f"Processing '{args.input}'...")
    t0 = time.time()
    
    try:
        # We'll use the fast Pillow version for the CLI tool
        result_pillow = dither_pillow(args.input, CHOSEN_PALETTE)
        result_pillow.save(args.output)
        print(f"Success! Dithered image saved to '{args.output}' (took {time.time()-t0:.3f}s)")
    except FileNotFoundError:
        print(f"Error: Could not find input image at '{args.input}'")
    except Exception as e:
        print(f"An error occurred: {e}")