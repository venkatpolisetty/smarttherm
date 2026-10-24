using System;
using System.IO;
using System.Threading;
using System.Diagnostics;
using System.Collections.Generic;
using System.Linq;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Text.Json;
using LibreHardwareMonitor.Hardware;
using Hwinfo.SharedMemory;

class Program
{
    private const long MaxTrainingFileBytes = 10 * 1024 * 1024;
    private const long MaxTrainingDataBytes = 512 * 1024 * 1024;
    private const int DatasetPruneSamples = 60;
    private const string TrainingCsvHeader =
        "schemaVersion,sessionId,timestamp,cpuTemperatureC,gpuTemperatureC," +
        "cpuUtilizationPercent,cpuClockMhz,cpuPowerW,cpuFanSpeedRpm,gpuUtilizationPercent," +
        "memoryUtilizationPercent,gpuSensor";
    private static volatile bool keepRunning;

    // Windows API: identify the process owning the foreground window.
    [DllImport("user32.dll")]
    private static extern IntPtr GetForegroundWindow();

    [DllImport("user32.dll")]
    private static extern uint GetWindowThreadProcessId(
        IntPtr hWnd,
        out uint processId);

    private sealed class ProcessSample
    {
        public int ProcessId { get; set; }
        public string ProcessName { get; set; } = "";
        public double CpuUsage { get; set; }
        public double RamUsageMb { get; set; }
        public bool IsForeground { get; set; }
        public string ProcessPath { get; set; } = "";
    }

    private static void Main()
    {
        using var collectorMutex = new Mutex(
            true, @"Local\SmartThermCollector", out bool createdNew);
        if (!createdNew)
        {
            Console.WriteLine("Another SmartTherm sensor collector is already running.");
            return;
        }

        keepRunning = true;
        // Give every run its own pair of CSV files.
        string sessionId = DateTime.Now.ToString(
            "yyyyMMdd_HHmmss_fff", CultureInfo.InvariantCulture);

        string systemFile = Path.Combine(
            Environment.CurrentDirectory, $"telemetry_{sessionId}.csv");

        string processFile = Path.Combine(
            Environment.CurrentDirectory, $"process_telemetry_{sessionId}.csv");
        string collectorDirectory = Path.Combine(Environment.CurrentDirectory, "ThermalCollector");
        if (!Directory.Exists(collectorDirectory))
            collectorDirectory = Environment.CurrentDirectory;
        string dataDirectory = Path.Combine(collectorDirectory, "data");
        Directory.CreateDirectory(dataDirectory);
        string currentReadingFile = Path.Combine(dataDirectory, "current-reading.json");
        string readingHistoryFile = Path.Combine(dataDirectory, "reading-history.jsonl");
        string trainingSessionId = Guid.NewGuid().ToString("N");
        StreamWriter? trainingWriter = null;
        int trainingFilePart = 0;
        int trainingSampleCount = 0;

        var computer = new Computer
        {
            IsCpuEnabled = true,
            IsMemoryEnabled = true,
            IsGpuEnabled = true,
            IsNetworkEnabled = true
        };

        // HWiNFO provides CPU and integrated-GPU temperatures that may not be
        // exposed by LibreHardwareMonitor on this laptop.
        using var hwinfoReader = new SharedMemoryReader();

        Console.CancelKeyPress += (_, e) =>
        {
            // Let the loop exit cleanly when the user presses Ctrl+C.
            e.Cancel = true;
            keepRunning = false;
        };
        _ = Task.Run(() =>
        {
            while (Console.ReadLine() is { } command)
            {
                if (command.Equals("stop", StringComparison.OrdinalIgnoreCase))
                {
                    keepRunning = false;
                    break;
                }
            }
        });

        try
        {
            bool hardwareReady = false;
            while (keepRunning)
            {
                try
                {
                    computer.Open();
                    hardwareReady = true;
                    break;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Hardware initialization warning: {ex.Message}");
                    Thread.Sleep(1000);
                }
            }

            // Print the sensors actually exposed by LibreHardwareMonitor.
            // This diagnostic output helps identify CPU/GPU sensor-name mismatches.
            Console.WriteLine();
            Console.WriteLine("===== HARDWARE SENSOR DIAGNOSTICS =====");
            foreach (IHardware hardware in computer.Hardware)
            {
                PrintDiagnosticSensors(hardware);
            }
            Console.WriteLine("===== END SENSOR DIAGNOSTICS =====");
            Console.WriteLine();

            // CPU_Temp is added. It will be blank if no CPU temperature
            // sensor is exposed by the hardware/LibreHardwareMonitor.
            File.WriteAllText(
                systemFile,
                "Timestamp,CPU_Load,CPU_Max_Load,CPU_Temp,CPU_Power_W," +
                "CPU_Fan_Speed_RPM,RAM_Load," +
                "Intel_GPU_Load,Intel_GPU_Temp,NVIDIA_GPU_Load,NVIDIA_GPU_Temp," +
                "GPU_Hotspot,GPU_Memory_Temp,VRAM_Used,GPU_Power," +
                "Upload_Speed,Download_Speed" +
                Environment.NewLine);

            File.WriteAllText(
                processFile,
                "Timestamp,ProcessId,ProcessName,CPU_Usage," +
                "RAM_Usage_MB,IsForeground,ProcessPath" +
                Environment.NewLine);

            Console.WriteLine("Telemetry collection started.");
            Console.WriteLine($"System data:  {systemFile}");
            Console.WriteLine($"Process data: {processFile}");
            Console.WriteLine("Press Ctrl+C to stop this session.");
            Console.WriteLine("HWiNFO temperatures are used when available; keep HWiNFO running.");
            Console.WriteLine("Intel GPU temperature will be blank if HWiNFO does not publish a matching sensor.");
            Console.WriteLine($"SmartTherm feed: {currentReadingFile}");
            Console.WriteLine($"Training telemetry: training_telemetry_{trainingSessionId}_part000.csv");
            Console.WriteLine();
            PruneTrainingFiles(dataDirectory, trainingSessionId, trainingFilePart);
            trainingWriter = OpenTrainingFile(dataDirectory, trainingSessionId, trainingFilePart);

            Dictionary<int, TimeSpan> previousCpuTimes =
                new Dictionary<int, TimeSpan>();

            Stopwatch stopwatch = Stopwatch.StartNew();
            double previousTime = stopwatch.Elapsed.TotalSeconds;
            bool processBaselineReady = false;

            while (keepRunning)
            {
                if (!hardwareReady)
                {
                    try
                    {
                        computer.Open();
                        hardwareReady = true;
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Hardware reconnect warning: {ex.Message}");
                        Thread.Sleep(1000);
                        continue;
                    }
                }

                float cpuLoad = float.NaN;
                float cpuMaxLoad = float.NaN;
                float ramLoad = float.NaN;

                // NaN represents an unavailable temperature, not a real 0°C.
                float cpuPackageTemp = float.NaN;
                float cpuMaxObservedTemp = float.NaN;
                float cpuClockMhz = float.NaN;
                float cpuPowerW = float.NaN;
                float cpuFanSpeedRpm = float.NaN;

                float intelGpuLoad = float.NaN;
                float intelGpuTemp = float.NaN;
                float gpuLoad = float.NaN; // NVIDIA GPU load
                float gpuTemp = float.NaN; // NVIDIA GPU temperature
                float gpuHotspot = float.NaN;
                float gpuMemoryTemp = float.NaN;
                float vramUsed = float.NaN;
                float gpuPower = float.NaN;
                float uploadSpeed = float.NaN;
                float downloadSpeed = float.NaN;

                bool sensorReadFailed = false;
                foreach (IHardware hardware in computer.Hardware)
                {
                    try
                    {
                        hardware.Update();
                        foreach (IHardware subHardware in hardware.SubHardware)
                            subHardware.Update();
                    }
                    catch (Exception ex)
                    {
                        sensorReadFailed = true;
                        Console.WriteLine($"Sensor update warning for {hardware.Name}: {ex.Message}");
                    }
                }

                // Read sensors from hardware and its sub-hardware.
                foreach (IHardware hardware in computer.Hardware)
                {
                    try
                    {
                        ReadHardwareSensors(
                            hardware,
                            ref cpuLoad,
                            ref cpuMaxLoad,
                            ref cpuPackageTemp,
                            ref cpuMaxObservedTemp,
                            ref cpuClockMhz,
                            ref cpuPowerW,
                            ref cpuFanSpeedRpm,
                            ref ramLoad,
                            ref intelGpuLoad,
                            ref intelGpuTemp,
                            ref gpuLoad,
                            ref gpuTemp,
                            ref gpuHotspot,
                            ref gpuMemoryTemp,
                            ref vramUsed,
                            ref gpuPower,
                            ref uploadSpeed,
                            ref downloadSpeed);
                    }
                    catch (Exception ex)
                    {
                        sensorReadFailed = true;
                        Console.WriteLine($"Sensor read warning for {hardware.Name}: {ex.Message}");
                    }

                    foreach (IHardware subHardware in hardware.SubHardware)
                    {
                        try
                        {
                            ReadHardwareSensors(
                                subHardware,
                                ref cpuLoad,
                                ref cpuMaxLoad,
                                ref cpuPackageTemp,
                                ref cpuMaxObservedTemp,
                                ref cpuClockMhz,
                                ref cpuPowerW,
                                ref cpuFanSpeedRpm,
                                ref ramLoad,
                                ref intelGpuLoad,
                                ref intelGpuTemp,
                                ref gpuLoad,
                                ref gpuTemp,
                                ref gpuHotspot,
                                ref gpuMemoryTemp,
                                ref vramUsed,
                                ref gpuPower,
                                ref uploadSpeed,
                                ref downloadSpeed);
                        }
                        catch (Exception ex)
                        {
                            sensorReadFailed = true;
                            Console.WriteLine($"Sensor read warning for {subHardware.Name}: {ex.Message}");
                        }
                    }
                }

                if (sensorReadFailed)
                {
                    try
                    {
                        computer.Close();
                        computer.Open();
                        hardwareReady = true;
                    }
                    catch (Exception ex)
                    {
                        hardwareReady = false;
                        Console.WriteLine($"Hardware reconnect warning: {ex.Message}");
                    }
                }

                float cpuTemp = !float.IsNaN(cpuPackageTemp)
                    ? cpuPackageTemp
                    : cpuMaxObservedTemp;

                // Read HWiNFO shared-memory sensors. These can provide CPU and
                // Intel integrated-GPU temperatures unavailable through LHM.
                TryReadHwinfoTemperatures(
                    hwinfoReader,
                    ref cpuTemp,
                    ref intelGpuTemp);

                float selectedGpuTemp = !float.IsNaN(gpuTemp)
                    ? gpuTemp
                    : intelGpuTemp;
                float selectedGpuLoad = !float.IsNaN(gpuLoad)
                    ? gpuLoad
                    : intelGpuLoad;
                bool sourceAvailable =
                    ValidTemperature(cpuTemp)
                    || ValidTemperature(selectedGpuTemp)
                    || ValidPercent(cpuLoad)
                    || ValidClock(cpuClockMhz)
                    || ValidPower(cpuPowerW)
                    || ValidFanSpeed(cpuFanSpeedRpm)
                    || ValidPercent(selectedGpuLoad)
                    || ValidPercent(ramLoad);
                if (!sourceAvailable)
                {
                    sensorReadFailed = true;
                    Console.WriteLine("No valid sensor readings are available; attempting hardware reconnect.");
                }
                var reading = new
                {
                    schemaVersion = 1,
                    sessionId = trainingSessionId,
                    timestamp = DateTimeOffset.UtcNow,
                    sourceAvailable,
                    cpuTemperatureC = ValidTemperature(cpuTemp) ? cpuTemp : (float?)null,
                    gpuTemperatureC = ValidTemperature(selectedGpuTemp) ? selectedGpuTemp : (float?)null,
                    cpuUtilizationPercent = ValidPercent(cpuLoad) ? cpuLoad : (float?)null,
                    cpuClockMhz = ValidClock(cpuClockMhz) ? cpuClockMhz : (float?)null,
                    cpuPowerW = ValidPower(cpuPowerW) ? cpuPowerW : (float?)null,
                    cpuFanSpeedRpm = ValidFanSpeed(cpuFanSpeedRpm) ? cpuFanSpeedRpm : (float?)null,
                    gpuUtilizationPercent = ValidPercent(selectedGpuLoad) ? selectedGpuLoad : (float?)null,
                    memoryUtilizationPercent = ValidPercent(ramLoad) ? ramLoad : (float?)null
                };
                string serializedReading = JsonSerializer.Serialize(reading);
                string temporaryReadingFile = currentReadingFile + ".tmp";
                File.WriteAllText(temporaryReadingFile, serializedReading);
                File.Move(temporaryReadingFile, currentReadingFile, true);

                var historyInfo = new FileInfo(readingHistoryFile);
                if (historyInfo.Exists && historyInfo.Length > 5 * 1024 * 1024)
                    File.WriteAllText(readingHistoryFile, "");
                File.AppendAllText(
                    readingHistoryFile, serializedReading + Environment.NewLine);
                AppendTrainingSample(
                    dataDirectory,
                    trainingSessionId,
                    ref trainingFilePart,
                    ref trainingWriter,
                    reading.timestamp,
                    cpuTemp,
                    selectedGpuTemp,
                    cpuLoad,
                    cpuClockMhz,
                    cpuPowerW,
                    cpuFanSpeedRpm,
                    selectedGpuLoad,
                    ramLoad,
                    ValidTemperature(gpuTemp) ? "NVIDIA" :
                        ValidTemperature(intelGpuTemp) ? "Intel" : "");
                if (++trainingSampleCount % DatasetPruneSamples == 0)
                {
                    PruneTrainingFiles(dataDirectory, trainingSessionId, trainingFilePart);
                }

                string timestamp = DateTime.Now.ToString(
                    "yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture);

                string systemRow = string.Join(",",
                    timestamp,
                    F(cpuLoad),
                    F(cpuMaxLoad),
                    FOrBlank(cpuTemp),
                    F(cpuPowerW),
                    F(cpuFanSpeedRpm),
                    F(ramLoad),
                    F(intelGpuLoad),
                    FOrBlank(intelGpuTemp),
                    F(gpuLoad),
                    FOrBlank(gpuTemp),
                    FOrBlank(gpuHotspot),
                    FOrBlank(gpuMemoryTemp),
                    F(vramUsed),
                    F(gpuPower),
                    F(uploadSpeed),
                    F(downloadSpeed));

                File.AppendAllText(
                    systemFile, systemRow + Environment.NewLine);

                Console.WriteLine(
                    $"SYSTEM | CPU {F(cpuLoad)}% | CPU Temp {FOrBlank(cpuTemp)}°C | " +
                    $"RAM {F(ramLoad)}% | Intel GPU {F(intelGpuLoad)}% / " +
                    $"{FOrBlank(intelGpuTemp)}°C | NVIDIA GPU {F(gpuLoad)}% / " +
                    $"{FOrBlank(gpuTemp)}°C");

                // Calculate elapsed time for per-process CPU measurements.
                double currentTime = stopwatch.Elapsed.TotalSeconds;
                double elapsedSeconds = currentTime - previousTime;
                if (elapsedSeconds <= 0)
                    elapsedSeconds = 5;
                previousTime = currentTime;

                var processSamples = new List<ProcessSample>();
                var activeProcessIds = new HashSet<int>();

                int foregroundProcessId = -1;
                try
                {
                    IntPtr foregroundWindow = GetForegroundWindow();
                    if (foregroundWindow != IntPtr.Zero)
                    {
                        GetWindowThreadProcessId(foregroundWindow, out uint pid);
                        foregroundProcessId = (int)pid;
                    }
                }
                catch
                {
                    // Foreground detection is optional.
                }

                foreach (Process process in Process.GetProcesses())
                {
                    try
                    {
                        activeProcessIds.Add(process.Id);
                        process.Refresh();

                        TimeSpan currentCpuTime = process.TotalProcessorTime;
                        double cpuUsage = 0;

                        if (previousCpuTimes.TryGetValue(
                            process.Id, out TimeSpan previousCpuTime))
                        {
                            double cpuSeconds =
                                (currentCpuTime - previousCpuTime).TotalSeconds;

                            cpuUsage = (cpuSeconds / elapsedSeconds /
                                Environment.ProcessorCount) * 100.0;

                            if (cpuUsage < 0)
                                cpuUsage = 0;
                        }

                        previousCpuTimes[process.Id] = currentCpuTime;

                        double ramMb = process.WorkingSet64 /
                            (1024.0 * 1024.0);

                        string path = "";
                        try
                        {
                            path = process.MainModule?.FileName ?? "";
                        }
                        catch
                        {
                            // Some processes don't allow path access.
                        }

                        processSamples.Add(new ProcessSample
                        {
                            ProcessId = process.Id,
                            ProcessName = process.ProcessName,
                            CpuUsage = cpuUsage,
                            RamUsageMb = ramMb,
                            IsForeground = process.Id == foregroundProcessId,
                            ProcessPath = path
                        });
                    }
                    catch
                    {
                        // Ignore processes that exit or deny access mid-sample.
                    }
                    finally
                    {
                        process.Dispose();
                    }
                }

                // Remove old PIDs so the history dictionary doesn't grow forever.
                foreach (int oldId in previousCpuTimes.Keys
                    .Where(id => !activeProcessIds.Contains(id)).ToList())
                {
                    previousCpuTimes.Remove(oldId);
                }

                if (!processBaselineReady)
                {
                    processBaselineReady = true;
                    Console.WriteLine(
                        "Process CPU baseline captured; measurements begin next interval.");
                }
                else
                {
                    var topProcesses = processSamples
                        .Where(p => p.CpuUsage > 0.1)
                        .OrderByDescending(p => p.CpuUsage)
                        .Take(10)
                        .ToList();

                    Console.WriteLine("========== TOP PROCESSES ==========");
                    if (topProcesses.Count == 0)
                    {
                        Console.WriteLine(
                            "No process currently using significant CPU.");
                    }
                    else
                    {
                        foreach (ProcessSample p in topProcesses)
                        {
                            string marker = p.IsForeground ? "*" : " ";
                            Console.WriteLine(
                                $"{marker} {p.ProcessName,-25} PID:{p.ProcessId,-7} " +
                                $"CPU:{p.CpuUsage,6:F2}% RAM:{p.RamUsageMb,8:F1} MB");
                        }
                    }
                    Console.WriteLine("===================================");

                    foreach (ProcessSample p in topProcesses)
                    {
                        // Keep CSV fields simple by replacing commas in text fields.
                        string safeName = p.ProcessName.Replace(",", " ");
                        string safePath = p.ProcessPath.Replace(",", " ");

                        string processRow = string.Join(",",
                            timestamp,
                            p.ProcessId.ToString(CultureInfo.InvariantCulture),
                            safeName,
                            F(p.CpuUsage),
                            F(p.RamUsageMb),
                            p.IsForeground.ToString(),
                            safePath);

                        File.AppendAllText(
                            processFile, processRow + Environment.NewLine);
                    }
                }

                // Sleep in short chunks so Ctrl+C can stop the session promptly.
                for (int i = 0; i < 50 && keepRunning; i++)
                    Thread.Sleep(100);
            }
        }
        finally
        {
            trainingWriter?.Dispose();
            computer.Close();
            Console.WriteLine();
            Console.WriteLine("Collection stopped.");
            Console.WriteLine($"Saved system data:  {systemFile}");
            Console.WriteLine($"Saved process data: {processFile}");
        }
    }

    private static void PrintDiagnosticSensors(IHardware hardware)
    {
        try
        {
            hardware.Update();

            foreach (ISensor sensor in hardware.Sensors)
            {
                if (sensor.SensorType != LibreHardwareMonitor.Hardware.SensorType.Temperature &&
                    sensor.SensorType != LibreHardwareMonitor.Hardware.SensorType.Load)
                    continue;

                string value = sensor.Value.HasValue
                    ? sensor.Value.Value.ToString("F2", CultureInfo.InvariantCulture)
                    : "unavailable";

                Console.WriteLine(
                    $"[SENSOR] Hardware={hardware.Name} | " +
                    $"HardwareType={hardware.HardwareType} | " +
                    $"Type={sensor.SensorType} | Name={sensor.Name} | Value={value}");
            }

            foreach (IHardware subHardware in hardware.SubHardware)
            {
                PrintDiagnosticSensors(subHardware);
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine(
                $"[SENSOR] Could not inspect {hardware.Name}: {ex.Message}");
        }
    }

    private static void ReadHardwareSensors(
        IHardware hardware,
        ref float cpuLoad,
        ref float cpuMaxLoad,
        ref float cpuPackageTemp,
        ref float cpuMaxObservedTemp,
        ref float cpuClockMhz,
        ref float cpuPowerW,
        ref float cpuFanSpeedRpm,
        ref float ramLoad,
        ref float intelGpuLoad,
        ref float intelGpuTemp,
        ref float gpuLoad,
        ref float gpuTemp,
        ref float gpuHotspot,
        ref float gpuMemoryTemp,
        ref float vramUsed,
        ref float gpuPower,
        ref float uploadSpeed,
        ref float downloadSpeed)
    {
        foreach (ISensor sensor in hardware.Sensors)
        {
            if (!sensor.Value.HasValue)
                continue;

            string name = sensor.Name ?? "";
            float value = sensor.Value.Value;

            if (hardware.HardwareType == HardwareType.Cpu)
            {
                if (sensor.SensorType == LibreHardwareMonitor.Hardware.SensorType.Power &&
                    (name.Contains("Package", StringComparison.OrdinalIgnoreCase) ||
                     name.Contains("CPU", StringComparison.OrdinalIgnoreCase)))
                    cpuPowerW = value;

                if (name == "CPU Total" &&
                    sensor.SensorType == LibreHardwareMonitor.Hardware.SensorType.Load)
                    cpuLoad = value;

                if (name == "CPU Core Max" &&
                    sensor.SensorType == LibreHardwareMonitor.Hardware.SensorType.Load)
                    cpuMaxLoad = value;

                if (sensor.SensorType == LibreHardwareMonitor.Hardware.SensorType.Temperature)
                {
                    // Prefer the CPU package reading when available.
                    if (name.Contains("CPU Package",
                        StringComparison.OrdinalIgnoreCase) ||
                        name.Equals("Package", StringComparison.OrdinalIgnoreCase))
                    {
                        cpuPackageTemp = value;
                    }

                    // Fallback: retain the highest CPU temperature sensor reading.
                    if (float.IsNaN(cpuMaxObservedTemp) ||
                        value > cpuMaxObservedTemp)
                    {
                        cpuMaxObservedTemp = value;
                    }
                }

                if (sensor.SensorType == LibreHardwareMonitor.Hardware.SensorType.Fan &&
                    name.Contains("CPU", StringComparison.OrdinalIgnoreCase))
                    cpuFanSpeedRpm = value;

                if (sensor.SensorType == LibreHardwareMonitor.Hardware.SensorType.Clock &&
                    (name.Contains("Core", StringComparison.OrdinalIgnoreCase) ||
                     name.Contains("CPU", StringComparison.OrdinalIgnoreCase)) &&
                    (float.IsNaN(cpuClockMhz) || value > cpuClockMhz))
                {
                    cpuClockMhz = value;
                }
            }

            if (hardware.HardwareType == HardwareType.Memory &&
                name == "Memory" &&
                sensor.SensorType == LibreHardwareMonitor.Hardware.SensorType.Load)
                ramLoad = value;

            if (hardware.HardwareType == HardwareType.GpuIntel)
            {
                if (sensor.SensorType == LibreHardwareMonitor.Hardware.SensorType.Load &&
                    (name.Equals("D3D 3D", StringComparison.OrdinalIgnoreCase) ||
                     name.Equals("GPU Core", StringComparison.OrdinalIgnoreCase)))
                {
                    intelGpuLoad = value;
                }

                if (sensor.SensorType == LibreHardwareMonitor.Hardware.SensorType.Temperature &&
                    (name.Contains("GPU", StringComparison.OrdinalIgnoreCase) ||
                     name.Contains("Graphics", StringComparison.OrdinalIgnoreCase) ||
                     name.Contains("GT", StringComparison.OrdinalIgnoreCase)))
                {
                    intelGpuTemp = value;
                }
            }

            if (hardware.HardwareType == HardwareType.GpuNvidia)
            {
                if (sensor.SensorType == LibreHardwareMonitor.Hardware.SensorType.Temperature &&
                    name == "GPU Core")
                    gpuTemp = value;

                if (sensor.SensorType == LibreHardwareMonitor.Hardware.SensorType.Load &&
                    name == "GPU Core")
                    gpuLoad = value;

                if (name == "GPU Hot Spot" &&
                    sensor.SensorType == LibreHardwareMonitor.Hardware.SensorType.Temperature)
                    gpuHotspot = value;

                if (name == "GPU Memory Junction" &&
                    sensor.SensorType == LibreHardwareMonitor.Hardware.SensorType.Temperature)
                    gpuMemoryTemp = value;

                if (name == "GPU Memory Used")
                    vramUsed = value;

                if (name == "GPU Package")
                    gpuPower = value;
            }

            if (hardware.HardwareType == HardwareType.Network)
            {
                if (name == "Upload Speed")
                    uploadSpeed = value;

                if (name == "Download Speed")
                    downloadSpeed = value;
            }
        }
    }

    private static void TryReadHwinfoTemperatures(
        SharedMemoryReader reader,
        ref float cpuTemp,
        ref float intelGpuTemp)
    {
        try
        {
            if (!reader.TryReadLocal(out var result))
                return;

            foreach (var reading in result.Readings)
            {
                string sensorName = reading.Sensor?.NameUser ?? "";
                string label = reading.LabelUser ?? "";
                string combined = sensorName + " " + label;
                string unit = reading.Unit ?? "";

                // Only temperature readings in Celsius are considered.
                if (!unit.Contains("°C", StringComparison.OrdinalIgnoreCase) &&
                    !unit.Contains("C", StringComparison.OrdinalIgnoreCase))
                    continue;

                if (!float.TryParse(
                    Convert.ToString(reading.Value, CultureInfo.InvariantCulture),
                    NumberStyles.Float,
                    CultureInfo.InvariantCulture,
                    out float value) ||
                    float.IsNaN(value) || float.IsInfinity(value) ||
                    value < 0 || value > 150)
                    continue;

                // Prefer HWiNFO's CPU package temperature.
                if (combined.Contains("CPU Package", StringComparison.OrdinalIgnoreCase) ||
                    label.Equals("CPU (Tctl/Tdie)", StringComparison.OrdinalIgnoreCase))
                {
                    cpuTemp = value;
                }

                // Intel iGPU temperature can be exposed as GT Cores / Graphics.
                // Do not use generic "GPU Temperature" here because that could
                // accidentally match the NVIDIA discrete GPU.
                bool mentionsIntelGraphics =
                    combined.Contains("Intel UHD", StringComparison.OrdinalIgnoreCase) ||
                    combined.Contains("GT Cores", StringComparison.OrdinalIgnoreCase) ||
                    combined.Contains("Graphics Temperature", StringComparison.OrdinalIgnoreCase) ||
                    combined.Contains("GPU GT", StringComparison.OrdinalIgnoreCase);

                if (mentionsIntelGraphics)
                    intelGpuTemp = value;
            }
        }
        catch (Exception ex)
        {
            // HWiNFO is optional at runtime; continue collecting LHM telemetry.
            Console.WriteLine($"HWiNFO read warning: {ex.Message}");
        }
    }

    private static string F(float value) =>
        ValidMetric(value) ? value.ToString("F2", CultureInfo.InvariantCulture) : "";

    private static string F(double value) =>
        value.ToString("F2", CultureInfo.InvariantCulture);

    private static string FOrBlank(float value) => F(value);

    private static StreamWriter OpenTrainingFile(
        string dataDirectory,
        string sessionId,
        int part)
    {
        string path = TrainingFilePath(dataDirectory, sessionId, part);
        bool writeHeader = !File.Exists(path) || new FileInfo(path).Length == 0;
        var writer = new StreamWriter(path, append: true, new System.Text.UTF8Encoding(false))
        {
            AutoFlush = true
        };
        if (writeHeader)
        {
            writer.WriteLine(TrainingCsvHeader);
        }
        return writer;
    }

    private static void AppendTrainingSample(
        string dataDirectory,
        string sessionId,
        ref int part,
        ref StreamWriter writer,
        DateTimeOffset timestamp,
        float cpuTemperature,
        float gpuTemperature,
        float cpuUtilization,
        float cpuClockMhz,
        float cpuPowerW,
        float cpuFanSpeedRpm,
        float gpuUtilization,
        float memoryUtilization,
        string gpuSensor)
    {
        if (writer.BaseStream.Length >= MaxTrainingFileBytes)
        {
            writer.Dispose();
            part++;
            writer = OpenTrainingFile(dataDirectory, sessionId, part);
            PruneTrainingFiles(dataDirectory, sessionId, part);
        }

        string[] fields =
        [
            "1",
            sessionId,
            timestamp.ToString("O", CultureInfo.InvariantCulture),
            TrainingMetric(cpuTemperature, ValidTemperature(cpuTemperature)),
            TrainingMetric(gpuTemperature, ValidTemperature(gpuTemperature)),
            TrainingMetric(cpuUtilization, ValidPercent(cpuUtilization)),
            TrainingMetric(cpuClockMhz, ValidClock(cpuClockMhz)),
            TrainingMetric(cpuPowerW, ValidPower(cpuPowerW)),
            TrainingMetric(cpuFanSpeedRpm, ValidFanSpeed(cpuFanSpeedRpm)),
            TrainingMetric(gpuUtilization, ValidPercent(gpuUtilization)),
            TrainingMetric(memoryUtilization, ValidPercent(memoryUtilization)),
            gpuSensor
        ];
        writer.WriteLine(string.Join(",", fields.Select(CsvField)));
    }

    private static void PruneTrainingFiles(
        string dataDirectory,
        string activeSessionId,
        int activePart)
    {
        string activePath = Path.GetFullPath(TrainingFilePath(
            dataDirectory, activeSessionId, activePart));
        var files = new DirectoryInfo(dataDirectory)
            .GetFiles("training_telemetry_*.csv")
            .Where(file => !string.Equals(
                Path.GetFullPath(file.FullName), activePath, StringComparison.OrdinalIgnoreCase))
            .OrderBy(file => file.LastWriteTimeUtc)
            .ToList();
        long totalBytes = files.Sum(file => file.Length);
        foreach (FileInfo file in files)
        {
            if (totalBytes <= MaxTrainingDataBytes)
            {
                break;
            }
            try
            {
                long size = file.Length;
                file.Delete();
                totalBytes -= size;
            }
            catch (IOException exception)
            {
                Console.WriteLine(
                    $"Training-data retention warning for {file.Name}: {exception.Message}");
            }
        }
    }

    private static string TrainingFilePath(
        string dataDirectory,
        string sessionId,
        int part) =>
        Path.Combine(dataDirectory, $"training_telemetry_{sessionId}_part{part:D3}.csv");

    private static string CsvField(string value) =>
        "\"" + value.Replace("\"", "\"\"", StringComparison.Ordinal) + "\"";

    private static string TrainingMetric(float value, bool valid) =>
        valid ? F(value) : "";

    private static bool ValidMetric(float value) =>
        !float.IsNaN(value) && !float.IsInfinity(value);

    private static bool ValidTemperature(float value) =>
        ValidMetric(value) && value >= 0 && value <= 150;

    private static bool ValidPercent(float value) =>
        ValidMetric(value) && value >= 0 && value <= 100;

    private static bool ValidClock(float value) =>
        ValidMetric(value) && value > 0 && value <= 10_000;

    private static bool ValidPower(float value) =>
        ValidMetric(value) && value >= 0 && value <= 1_000;

    private static bool ValidFanSpeed(float value) =>
        ValidMetric(value) && value >= 0 && value <= 100_000;
}