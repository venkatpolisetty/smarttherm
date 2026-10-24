package local.smarttherm.model;

import java.time.Instant;

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
        String sessionId) {
}
