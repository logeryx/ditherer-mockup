using CommandLine;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.ColorSpaces;
using SixLabors.ImageSharp.PixelFormats;
using System;
using System.IO; // Pridané pre prácu so súbormi
using System.Threading.Tasks; // Pre asynchrónne operácie s ImageSharp

namespace uloha2_ImagePallete;

public class Options
{
  [Option('o', "output", Required = false, HelpText = "Output file-name (PNG).")]
  public string OutputFileName { get; set; } = string.Empty;

  [Option('i', "input", Required = true, Default = "", HelpText = "Input image.")]
  public string InputFileName { get; set; } = string.Empty;

  [Option('c', "colors", Required = false, HelpText = "Required number of colors.")]
  public int? NumberOfColors { get; set; } = null;
}


class Program
{
  // Zmena na asynchrónnu Main metódu
  static async Task Main (string[] args)
  {
    await Parser.Default.ParseArguments<Options>(args)
        .WithParsedAsync(async options =>
        {
          if (!ParameterControler.InitialControl(options))
            return;

          await GeneratingModes.ModeCaller(options);
        });
  }
}

static public class ParameterControler
{
  static public bool InitialControl (Options options)
  {
    // Kontrola, či je obrázok PNG (zjednodušená)
    if (options.NumberOfColors < 1)
    {
      Console.WriteLine("ERROR : not enough colors");
      return false;
    }
    else if (options.NumberOfColors > 10)
    {
      Console.WriteLine("ERROR : too much colors");
      return false;
    } 

    if (!File.Exists(options.InputFileName))
    {
      Console.WriteLine($"ERROR: Input file not found: {options.InputFileName}");
      return false;
    }
    else
    {
      if (options.InputFileName.Length < 4 || !(options.InputFileName[^4..].ToLower() == ".png" || options.InputFileName[^4..].ToLower() == ".jpg"))
      {
        Console.WriteLine("ERROR : Input file must be .png or .jpg");
        return false;
      }
    }

    // POZNÁMKA: Kontrola koncovky výstupu by mala byť na OutputFileName, nie InputFileName
    if (options.OutputFileName != null)
    {
      if ((string.IsNullOrEmpty(options.OutputFileName)) == false)
      {
        if (options.OutputFileName.Length < 4 || options.OutputFileName[^4..].ToLower() != ".png")
        {
          Console.WriteLine("ERROR : Output file must be .png");
          return false;
        }
      }

    }
    return true;
  }
}

// Pridanie chýbajúcej triedy GeneretingModes
public static class GeneratingModes
{
    public static async Task ModeCaller(Options options)
    {
        try
        {
          // Načítanie obrázka v požadovanom formáte
          using (var originalImage = await Image.LoadAsync<Rgb24>(options.InputFileName))
          {
            //Console.WriteLine($"Extrahujem {options.NumberOfColors} hlavných farieb z {options.InputFileName}...");

            // Použijeme nový rýchly extractor + kreslenie farebných obdĺžnikov
            ColorPaletteExtractor.ExtractAndDrawPalette(originalImage, options.OutputFileName, options.NumberOfColors);
          }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"ERROR during image processing: {ex.Message}");
        }
    }
}

// ... (ďalší kód ImageQuantizerHsv nasleduje v sekcii 2)
