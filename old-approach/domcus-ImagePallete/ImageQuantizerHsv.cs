using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.ColorSpaces;
using SixLabors.ImageSharp.ColorSpaces.Conversion;
using System;
using System.Collections.Generic;
using System.Linq;

namespace uloha2_ImagePallete
{
  public static class ColorPaletteExtractor
  {
    public static void ExtractAndDrawPalette (Image<Rgb24> originalImage, string? outputFile = null, int? requestedColors = null)
    {
      // 1️⃣ Histogram farieb (5bit redukcia)
      var histogram = new Dictionary<(byte R, byte G, byte B), int>();

      for (int y = 0; y < originalImage.Height; y++)
      {
        for (int x = 0; x < originalImage.Width; x++)
        {
          var p = originalImage[x, y];
          var key = ((byte)(p.R >> 3), (byte)(p.G >> 3), (byte)(p.B >> 3));
          if (!histogram.ContainsKey(key))
            histogram[key] = 0;
          histogram[key]++;
        }
      }

      // 2️⃣ Median cut
      int targetCount = requestedColors ?? 10;
      var weightedColors = MedianCutOnHistogram(histogram, targetCount);

      // 3️⃣ Výber farieb
      List<Rgb24> finalColors;
      if (requestedColors.HasValue)
        finalColors = SelectDistinctColorsForFixedCount(weightedColors, requestedColors.Value);
      else
        finalColors = SelectDistinctColors(weightedColors);

      // 4️⃣ Spektrálne zoradenie podľa Hue + malá váha Value
      finalColors = SortColorsByHueAndValue(finalColors);

      // 5️⃣ Výstup – obrázok alebo konzola
      if (!string.IsNullOrEmpty(outputFile))
      {
        int rectHeight = Math.Max(40, originalImage.Height / 10);
        int width = originalImage.Width;
        int height = originalImage.Height + rectHeight;

        using var outputImage = new Image<Rgb24>(width, height);
        outputImage.Mutate(ctx => ctx.DrawImage(originalImage, new Point(0, 0), 1f));

        int rectWidth = width / finalColors.Count;

        for (int i = 0; i < finalColors.Count; i++)
        {
          int xStart = i * rectWidth;
          int xEnd = (i == finalColors.Count - 1) ? width : xStart + rectWidth;

          for (int x = xStart; x < xEnd; x++)
            for (int y = originalImage.Height; y < height; y++)
              outputImage[x, y] = finalColors[i];
        }

        outputImage.Save(outputFile);
        //Console.WriteLine($"Výsledok uložený do {outputFile}, farby: {finalColors.Count}");
      }
      else
      {
        //Console.WriteLine("Paleta farieb (RGB):");
        foreach (var c in finalColors) {
          Console.Write($"#{c.R:X2}{c.G:X2}{c.B:X2} ");
        }
          // Console.WriteLine($"{c.R},{c.G},{c.B}");
      }
    }

    // ------------------- Výber farieb -------------------
    private static List<Rgb24> SelectDistinctColors (List<(Rgb24 Color, int Weight)> weightedColors)
    {
      var converter = new ColorSpaceConverter();
      var sorted = weightedColors.OrderByDescending(c => c.Weight).ToList();
      var finalColors = new List<Rgb24>();
      double minDistance = 20;

      foreach (var c in sorted)
      {
        var labC = converter.ToCieLab(c.Color);

        bool tooSimilar = finalColors.Any(fc =>
        {
          var labFc = converter.ToCieLab(fc);
          return LabDistance(labC, labFc) < minDistance;
        });

        if (!tooSimilar)
          finalColors.Add(c.Color);
      }

      if (finalColors.Count < 3)
        finalColors = sorted.Take(3).Select(c => c.Color).ToList();

      if (finalColors.Count > 10)
        finalColors = finalColors.Take(10).ToList();

      return finalColors;
    }

    private static List<Rgb24> SelectDistinctColorsForFixedCount (List<(Rgb24 Color, int Weight)> weightedColors, int count)
    {
      var converter = new ColorSpaceConverter();
      var sorted = weightedColors.OrderByDescending(c => c.Weight).ToList();
      var finalColors = new List<Rgb24>();
      double minDistance = 20;

      foreach (var c in sorted)
      {
        var labC = converter.ToCieLab(c.Color);

        bool tooSimilar = finalColors.Any(fc =>
        {
          var labFc = converter.ToCieLab(fc);
          return LabDistance(labC, labFc) < minDistance;
        });

        if (!tooSimilar)
          finalColors.Add(c.Color);

        if (finalColors.Count == count)
          break;
      }

      int i = 0;
      while (finalColors.Count < count && i < sorted.Count)
      {
        if (!finalColors.Contains(sorted[i].Color))
          finalColors.Add(sorted[i].Color);
        i++;
      }

      return finalColors;
    }

    private static double LabDistance (CieLab a, CieLab b)
    {
      double dL = a.L - b.L;
      double da = a.A - b.A;
      double db = a.B - b.B;
      return Math.Sqrt(dL * dL + da * da + db * db);
    }

    // ------------------- Median cut -------------------
    private static List<(Rgb24 Color, int Weight)> MedianCutOnHistogram (Dictionary<(byte R, byte G, byte B), int> histogram, int targetCount)
    {
      var pixels = histogram.Select(kv => new RgbWeighted
      {
        Color = new Rgb24((byte)(kv.Key.R << 3), (byte)(kv.Key.G << 3), (byte)(kv.Key.B << 3)),
        Weight = kv.Value
      }).ToList();

      var boxes = new List<ColorBoxWeighted> { new ColorBoxWeighted(pixels) };

      while (boxes.Count < targetCount)
      {
        boxes.Sort((a, b) => b.Range.CompareTo(a.Range));
        var box = boxes[0];
        if (box.Pixels.Count <= 1)
          break;
        boxes.RemoveAt(0);
        boxes.AddRange(box.Split());
      }

      return boxes.Select(b => (b.AverageColor(), b.TotalWeight())).ToList();
    }

    private class RgbWeighted
    {
      public Rgb24 Color;
      public int Weight;
    }

    private class ColorBoxWeighted
    {
      public List<RgbWeighted> Pixels;
      public int Range { get; private set; }

      public ColorBoxWeighted (List<RgbWeighted> pixels)
      {
        Pixels = pixels;
        UpdateRange();
      }

      private void UpdateRange ()
      {
        byte rMin = Pixels.Min(p => p.Color.R), rMax = Pixels.Max(p => p.Color.R);
        byte gMin = Pixels.Min(p => p.Color.G), gMax = Pixels.Max(p => p.Color.G);
        byte bMin = Pixels.Min(p => p.Color.B), bMax = Pixels.Max(p => p.Color.B);
        Range = Math.Max(rMax - rMin, Math.Max(gMax - gMin, bMax - bMin));
      }

      public List<ColorBoxWeighted> Split ()
      {
        int rRange = Pixels.Max(p => p.Color.R) - Pixels.Min(p => p.Color.R);
        int gRange = Pixels.Max(p => p.Color.G) - Pixels.Min(p => p.Color.G);
        int bRange = Pixels.Max(p => p.Color.B) - Pixels.Min(p => p.Color.B);

        if (rRange >= gRange && rRange >= bRange)
          Pixels.Sort((a, b) => a.Color.R.CompareTo(b.Color.R));
        else if (gRange >= rRange && gRange >= bRange)
          Pixels.Sort((a, b) => a.Color.G.CompareTo(b.Color.G));
        else
          Pixels.Sort((a, b) => a.Color.B.CompareTo(b.Color.B));

        int mid = Pixels.Count / 2;
        var first = Pixels.Take(mid).ToList();
        var second = Pixels.Skip(mid).ToList();

        return new List<ColorBoxWeighted> { new ColorBoxWeighted(first), new ColorBoxWeighted(second) };
      }

      public int TotalWeight () => Pixels.Sum(p => p.Weight);

      public Rgb24 AverageColor ()
      {
        long sumR = 0, sumG = 0, sumB = 0, sumW = 0;

        foreach (var p in Pixels)
        {
          sumR += p.Color.R * p.Weight;
          sumG += p.Color.G * p.Weight;
          sumB += p.Color.B * p.Weight;
          sumW += p.Weight;
        }

        return new Rgb24((byte)(sumR / sumW), (byte)(sumG / sumW), (byte)(sumB / sumW));
      }
    }

    // ------------------- Spektrálne zoradenie -------------------
    private static List<Rgb24> SortColorsByHueAndValue (List<Rgb24> colors, double valueWeight = 0.15)
    {
      return colors
          .Select(c =>
          {
            var hsv = RgbToHsv(c);
            double score = hsv.H + hsv.V * valueWeight; // HLAVNÉ Zoradenie Hue + jemný Value
            return (Rgb: c, Score: score);
          })
          .OrderBy(x => x.Score)
          .Select(x => x.Rgb)
          .ToList();
    }

    private static (double H, double S, double V) RgbToHsv (Rgb24 color)
    {
      double r = color.R / 255.0;
      double g = color.G / 255.0;
      double b = color.B / 255.0;

      double max = Math.Max(r, Math.Max(g, b));
      double min = Math.Min(r, Math.Min(g, b));
      double delta = max - min;

      double h = 0;

      if (delta != 0)
      {
        if (max == r)
          h = 60 * (((g - b) / delta) % 6);
        else if (max == g)
          h = 60 * (((b - r) / delta) + 2);
        else
          h = 60 * (((r - g) / delta) + 4);
      }

      if (h < 0)
        h += 360;

      double s = (max == 0) ? 0 : delta / max;
      double v = max;

      return (h, s, v);
    }
  }
}
