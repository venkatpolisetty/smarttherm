using SmartTherm;
using System.Text.Json.Nodes;

static void Assert(bool condition, string message)
{
    if (!condition) throw new Exception(message);
}

double calculatedCpu = ProcessSampler.CalculateCpuPercent(
    TimeSpan.FromSeconds(1), elapsedSeconds: 2, processorCount: 4);
Assert(Math.Abs(calculatedCpu - 12.5) < 0.001,
    "CPU percent must divide process CPU seconds by elapsed seconds and processor count.");

List<ProcessUsage> aggregatedProcesses = ProcessSampler.AggregateByName([
    new ProcessUsage("chrome", 8.0),
    new ProcessUsage("chrome", 5.5),
    new ProcessUsage("python", 4.0)
]);
Assert(aggregatedProcesses.Count == 2 &&
       aggregatedProcesses[0].Name == "chrome" &&
       Math.Abs(aggregatedProcesses[0].CpuPercent - 13.5) < 0.001,
    "Processes with the same name must be aggregated and sorted by CPU percent.");

DateTimeOffset start = DateTimeOffset.UtcNow;
var analyzer = new ThrottleCalibrationAnalyzer();
for (int i = 0; i < 8; i++)
    analyzer.Add(new(start.AddSeconds(i * 5), 70, 95, 3000, 45, false));
for (int i = 8; i < 11; i++)
    analyzer.Add(new(start.AddSeconds(i * 5), 88 + (i - 8), 94, 2400, 35, true));
var confirmed = analyzer.Complete(start.AddMinutes(5));
Assert(confirmed.DetectionMethod == "hardware_confirmed", "Hardware flag must confirm the event.");
Assert(confirmed.EventCount == 1, "Three consecutive samples should be one event.");
Assert(confirmed.EstimatedTemperatureC == 89, "Event temperature should be the median.");

var suspectedAnalyzer = new ThrottleCalibrationAnalyzer();
for (int i = 0; i < 8; i++)
    suspectedAnalyzer.Add(new(start.AddSeconds(i * 5), 70, 95, 3000, 45, null));
for (int i = 8; i < 11; i++)
    suspectedAnalyzer.Add(new(start.AddSeconds(i * 5), 76, 94, 2400, null, null));
Assert(suspectedAnalyzer.Complete(start).DetectionMethod == "suspected_sensor_trend",
    "A relative temperature rise and clock drop without a hardware flag is suspected, not confirmed.");

var hardwareBelowEightyAnalyzer = new ThrottleCalibrationAnalyzer();
for (int i = 0; i < 3; i++)
    hardwareBelowEightyAnalyzer.Add(new(start.AddSeconds(i * 5), 70, 95, null, 40, false));
for (int i = 3; i < 6; i++)
    hardwareBelowEightyAnalyzer.Add(new(start.AddSeconds(i * 5), 76, 94, null, 35, true));
var hardwareBelowEighty = hardwareBelowEightyAnalyzer.Complete(start);
Assert(hardwareBelowEighty.DetectionMethod == "hardware_confirmed" &&
       hardwareBelowEighty.EstimatedTemperatureC == 76,
    "A hardware-confirmed event below 80°C must remain detectable without a clock sensor.");

var staleGapAnalyzer = new ThrottleCalibrationAnalyzer();
for (int i = 0; i < 6; i++)
    staleGapAnalyzer.Add(new(start.AddSeconds(i * 5), 70, 95, 3000, 45, false));
staleGapAnalyzer.Add(new(start.AddSeconds(30), 86, 94, 2400, 35, true));
staleGapAnalyzer.Add(new(start.AddSeconds(50), 86, 94, 2400, 35, true));
staleGapAnalyzer.Add(new(start.AddSeconds(55), 86, 94, 2400, 35, true));
Assert(staleGapAnalyzer.Complete(start).EstimatedTemperatureC is null,
    "A stale sample gap must break the consecutive event sequence.");

var noEvent = new ThrottleCalibrationAnalyzer();
noEvent.Add(new(start, 75, 95, 2000, 30, false));
Assert(noEvent.Complete(start).EstimatedTemperatureC is null,
    "No event must not fall back to the maximum temperature.");

string profilePath = Path.Combine(Path.GetTempPath(), $"smarttherm-profile-{Guid.NewGuid():N}.json");
try
{
    File.WriteAllText(profilePath, "{\"existingSetting\":\"preserve\"}");
    SmartThermProfile.Save(profilePath, confirmed);
    var profile = JsonNode.Parse(File.ReadAllText(profilePath))!;
    Assert(profile["existingSetting"]?.GetValue<string>() == "preserve",
        "Profile writes must preserve existing fields.");
    Assert(profile["cpuThrottleTemperatureC"]?.GetValue<double>() == 89,
        "Profile must persist the detected event temperature.");
}
finally
{
    File.Delete(profilePath);
}

Console.WriteLine("Throttle analyzer and process sampler unit checks passed.");
