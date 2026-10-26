package local.smarttherm.model;

import java.time.Instant;
import java.util.List;

public record ThermalReading(
        Instant timestamp,
        boolean sourceAvailable,
        Double cpuTemperatureC,
        Double gpuTemperatureC,
        Double cpuUtilizationPercent,
        Double cpuClockMhz,
        Double cpuPowerW,
        Double cpuFanSpeedRpm,
        Double gpuUtilizationPercent,
        Double memoryUtilizationPercent,
        String sessionId,
        List<TopProcessUsage> topProcesses) {

    public ThermalReading(
            Instant timestamp,
            boolean sourceAvailable,
            Double cpuTemperatureC,
            Double gpuTemperatureC,
            Double cpuUtilizationPercent,
            Double cpuClockMhz,
            Double cpuPowerW,
            Double cpuFanSpeedRpm,
            Double gpuUtilizationPercent,
            Double memoryUtilizationPercent,
            String sessionId) {
        this(timestamp, sourceAvailable, cpuTemperatureC, gpuTemperatureC,
                cpuUtilizationPercent, cpuClockMhz, cpuPowerW, cpuFanSpeedRpm,
                gpuUtilizationPercent, memoryUtilizationPercent, sessionId, null);
    }
}
