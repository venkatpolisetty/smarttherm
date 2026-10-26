using System.Text.Json;
using System.Text.Json.Nodes;

namespace SmartTherm;

public sealed record CpuThermalSample(
    DateTimeOffset Timestamp,
    double? TemperatureC,
    double? UtilizationPercent,
    double? ClockMhz,
    double? PackagePowerW,
    bool? HardwareThermalThrottle);

public sealed record ThrottleCalibrationResult(
    double? EstimatedTemperatureC,
    string DetectionMethod,
    int EventCount,
    DateTimeOffset CalibrationTimestamp,
    double? BaselineClockMhz,
    IReadOnlyList<double> EventTemperaturesC);

public sealed class ThrottleCalibrationAnalyzer
{
    private readonly List<CpuThermalSample> samples = [];

    public void Add(CpuThermalSample sample) => samples.Add(sample);

    public ThrottleCalibrationResult Complete(DateTimeOffset completedAt)
    {
        var valid = samples.Where(IsValid).OrderBy(s => s.Timestamp).ToList();
        var baselineSamples = valid.Where(s => s.UtilizationPercent.GetValueOrDefault() >= 80 && s.ClockMhz.HasValue)
            .Take(6).ToList();
        var baselineClocks = baselineSamples.Select(s => s.ClockMhz!.Value).Order().ToList();
        double? baselineTemperature = baselineSamples.Count >= 3
            ? Median(baselineSamples.Select(s => s.TemperatureC!.Value))
            : null;
        double? baseline = baselineClocks.Count >= 3
            ? baselineClocks[(int)Math.Floor((baselineClocks.Count - 1) * 0.8)]
            : null;

        var events = new List<(double Temp, bool Confirmed)>();
        int run = 0;
        var runSamples = new List<CpuThermalSample>();
        foreach (var sample in samples.OrderBy(s => s.Timestamp))
        {
            bool highLoad = IsValid(sample) && sample.UtilizationPercent.GetValueOrDefault() >= 80;
            bool hardwareConfirmed = highLoad && sample.HardwareThermalThrottle == true;
            bool suspectedFromTrend = highLoad && baseline.HasValue && baselineTemperature.HasValue &&
                sample.ClockMhz.HasValue && sample.ClockMhz.Value <= baseline * 0.85 &&
                sample.TemperatureC!.Value >= baselineTemperature.Value + 5;
            bool candidate = hardwareConfirmed || suspectedFromTrend;
            if (candidate)
            {
                if (runSamples.Count > 0 && sample.Timestamp - runSamples[^1].Timestamp > TimeSpan.FromSeconds(15))
                { run = 0; runSamples.Clear(); }
                run++;
                runSamples.Add(sample);
                if (run == 3)
                    events.Add((Median(runSamples.TakeLast(3).Select(s => s.TemperatureC!.Value)),
                        runSamples.TakeLast(3).Any(s => s.HardwareThermalThrottle == true)));
            }
            else { run = 0; runSamples.Clear(); }
        }

        var temperatures = events.Select(e => e.Temp).ToList();
        double? estimate = temperatures.Count == 0 ? null : Median(temperatures);
        string method = events.Any(e => e.Confirmed) ? "hardware_confirmed" :
            events.Count > 0 ? "suspected_sensor_trend" : "not_determined";
        return new(estimate, method, events.Count, completedAt, baseline,
            events.Select(e => e.Temp).ToArray());
    }

    public static bool IsValid(CpuThermalSample s) =>
        s.TemperatureC is >= 0 and <= 150 && s.UtilizationPercent is >= 0 and <= 100 &&
        (s.ClockMhz is null or (> 0 and <= 10000)) &&
        (s.PackagePowerW is null or (>= 0 and <= 1000));

    private static double Median(IEnumerable<double> values)
    {
        var ordered = values.Order().ToArray();
        return ordered.Length % 2 == 1 ? ordered[ordered.Length / 2] :
            (ordered[ordered.Length / 2 - 1] + ordered[ordered.Length / 2]) / 2;
    }
}

public static class SmartThermProfile
{
    public static void Save(string path, ThrottleCalibrationResult result)
    {
        JsonObject profile;
        try { profile = JsonNode.Parse(File.Exists(path) ? File.ReadAllText(path) : "{}") as JsonObject ?? new(); }
        catch (JsonException) { profile = new(); }
        profile["cpuThrottleTemperatureC"] = result.EstimatedTemperatureC is null
            ? null : JsonValue.Create(result.EstimatedTemperatureC.Value);
        profile["cpuThrottleDetectionMethod"] = result.DetectionMethod;
        profile["cpuThrottleEventCount"] = result.EventCount;
        profile["cpuThrottleCalibrationTimestamp"] = result.CalibrationTimestamp;
        profile["cpuThrottleBaselineClockMhz"] = result.BaselineClockMhz is null
            ? null : JsonValue.Create(result.BaselineClockMhz.Value);
        profile["cpuThrottleEventTemperaturesC"] = JsonSerializer.SerializeToNode(result.EventTemperaturesC);
        var temp = path + ".tmp";
        File.WriteAllText(temp, profile.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
        File.Move(temp, path, true);
    }
}
