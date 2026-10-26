using System.Diagnostics;

namespace SmartTherm;

public sealed record ProcessUsage(string Name, double CpuPercent);

/// <summary>Samples process CPU time using the collector's existing sampling cycle.</summary>
public sealed class ProcessSampler
{
    private readonly record struct ProcessKey(int ProcessId, long StartTimeTicks);
    private readonly record struct Snapshot(ProcessKey Key, string Name, TimeSpan CpuTime);

    private Dictionary<ProcessKey, TimeSpan> _previousCpuTimes = [];
    private Dictionary<string, double> _smoothedByName = new(StringComparer.OrdinalIgnoreCase);
    private long? _previousTimestamp;

    public List<ProcessUsage> Sample(int top = 5)
    {
        long startedAt = Stopwatch.GetTimestamp();
        try
        {
            var snapshots = ReadSnapshots();
            long sampledAt = Stopwatch.GetTimestamp();
            double elapsedSeconds = _previousTimestamp is long previousSampleTimestamp
                ? Stopwatch.GetElapsedTime(previousSampleTimestamp, sampledAt).TotalSeconds
                : 0;
            var currentCpuTimes = snapshots.ToDictionary(item => item.Key, item => item.CpuTime);

            if (_previousTimestamp is null || elapsedSeconds <= 0)
            {
                _previousCpuTimes = currentCpuTimes;
                _previousTimestamp = sampledAt;
                return [];
            }

            var measured = new List<ProcessUsage>();
            foreach (Snapshot snapshot in snapshots)
            {
                if (!_previousCpuTimes.TryGetValue(snapshot.Key, out TimeSpan previousCpuTime))
                    continue;

                TimeSpan delta = snapshot.CpuTime - previousCpuTime;
                if (delta < TimeSpan.Zero)
                    continue;
                measured.Add(new ProcessUsage(snapshot.Name,
                    CalculateCpuPercent(delta, elapsedSeconds, Environment.ProcessorCount)));
            }

            List<ProcessUsage> aggregate = AggregateByName(measured);
            var currentByName = aggregate.ToDictionary(item => item.Name, item => item.CpuPercent,
                StringComparer.OrdinalIgnoreCase);
            foreach (string name in _smoothedByName.Keys.Union(currentByName.Keys,
                         StringComparer.OrdinalIgnoreCase).ToList())
            {
                double previous = _smoothedByName.GetValueOrDefault(name);
                double current = currentByName.GetValueOrDefault(name);
                double smoothed = 0.6 * previous + 0.4 * current;
                if (smoothed < 0.05)
                    _smoothedByName.Remove(name);
                else
                    _smoothedByName[name] = smoothed;
            }

            _previousCpuTimes = currentCpuTimes;
            _previousTimestamp = sampledAt;
            return _smoothedByName
                .Select(item => new ProcessUsage(item.Key, item.Value))
                .OrderByDescending(item => item.CpuPercent)
                .Take(Math.Max(0, top))
                .ToList();
        }
        finally
        {
            Debug.WriteLine($"[DEBUG] ProcessSampler.Sample completed in {Stopwatch.GetElapsedTime(startedAt).TotalMilliseconds:F1} ms.");
        }
    }

    public static double CalculateCpuPercent(TimeSpan deltaCpu, double elapsedSeconds, int processorCount)
    {
        if (elapsedSeconds <= 0 || processorCount <= 0 || deltaCpu <= TimeSpan.Zero)
            return 0;
        return Math.Clamp(deltaCpu.TotalSeconds / (elapsedSeconds * processorCount) * 100.0, 0, 100);
    }

    public static List<ProcessUsage> AggregateByName(IEnumerable<ProcessUsage> samples) => samples
        .Where(item => !string.IsNullOrWhiteSpace(item.Name) && double.IsFinite(item.CpuPercent))
        .GroupBy(item => item.Name, StringComparer.OrdinalIgnoreCase)
        .Select(group => new ProcessUsage(group.Key, group.Sum(item => item.CpuPercent)))
        .OrderByDescending(item => item.CpuPercent)
        .ToList();

    private static List<Snapshot> ReadSnapshots()
    {
        var snapshots = new List<Snapshot>();
        Process[] processes;
        try
        {
            processes = Process.GetProcesses();
        }
        catch
        {
            return snapshots;
        }

        foreach (Process process in processes)
        {
            try
            {
                int pid = process.Id;
                if (pid == 0 || pid == Environment.ProcessId)
                    continue;

                DateTime startTime = process.StartTime;
                string name = process.ProcessName;
                TimeSpan cpuTime = process.TotalProcessorTime;
                snapshots.Add(new Snapshot(
                    new ProcessKey(pid, startTime.ToUniversalTime().Ticks), name, cpuTime));
            }
            catch
            {
                // Access denied or a process exiting during this snapshot is isolated.
            }
            finally
            {
                process.Dispose();
            }
        }

        return snapshots;
    }
}
