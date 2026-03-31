import os
import subprocess
import argparse
from pathlib import Path

# =====================================================================
# Configuration: Set this to how you execute your C# program!
# Examples:
# CSHARP_COMMAND = ["MedianCutApp.exe"] 
# CSHARP_COMMAND = ["dotnet", "run", "--project", "MedianCutApp", "--"]
# =====================================================================
CSHARP_COMMAND = ["./domcus-ImagePallete/bin/Debug/net8.0/uloha2-ImagePallete", "-c", "4", "-i"] 

def process_directory(input_dir, output_dir):
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    
    # Create the output directory if it doesn't exist
    out_path.mkdir(parents=True, exist_ok=True)

    # Valid image extensions to look for
    valid_exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    
    # Iterate through all files in the input directory
    for img_file in in_path.iterdir():
        if img_file.suffix.lower() in valid_exts:
            print(f"\n--- Processing {img_file.name} ---")
            
            # 1. Call C# program (passing the image path as an argument)
            # e.g., MedianCut.exe "input/image1.png"
            cmd_csharp = CSHARP_COMMAND + [str(img_file)]
            
            try:
                # Run C# app and capture its standard output
                print(cmd_csharp)
                result = subprocess.run(cmd_csharp, capture_output=True, text=True, check=True)
                
                # Clean up the output and split by spaces to get the hex codes
                palette_output = result.stdout.strip()
                palette_args = palette_output.split()
                
                # Basic validation to ensure it looks like hex codes
                if not palette_args or not palette_args[0].startswith('#'):
                    print(f"  [!] Warning: Unexpected output from C#: '{palette_output}'")
                    continue
                    
                print(f"  -> Extracted palette from C#: {palette_args}")
                
            except subprocess.CalledProcessError as e:
                print(f"  [!] Error running C# program on {img_file.name}: {e.stderr}")
                continue
            except FileNotFoundError:
                print(f"  [!] Error: Could not find C# executable '{CSHARP_COMMAND[0]}'. Check your CSHARP_COMMAND variable.")
                return

            # 2. Setup paths and call the Python dithering script
            # Constructs filename like: "image1-dither.png"
            out_file = out_path / f"{img_file.stem}_0dither.png"
            
            cmd_python = ["python", "palette_dither.py", str(img_file), str(out_file), "--palette"] + palette_args
            
            try:
                subprocess.run(cmd_python, check=True)
            except subprocess.CalledProcessError as e:
                print(f"  [!] Error running dither script on {img_file.name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process: C# Median Cut -> Python Dither")
    parser.add_argument("input_dir", help="Directory containing original images")
    parser.add_argument("output_dir", help="Directory to save dithered images")
    args = parser.parse_args()
    
    process_directory(args.input_dir, args.output_dir)
