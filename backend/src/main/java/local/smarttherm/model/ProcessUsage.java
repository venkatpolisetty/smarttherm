package local.smarttherm.model;

import java.time.Instant;
import java.util.List;

public record ProcessUsage(
        Instant timestamp,
        boolean available,
        String message,
        List<ProcessEntry> processes) {
    public record ProcessEntry(
            int processId,
            String processName,
            double cpuUsagePercent,
            double memoryMb,
            boolean foreground) {}
}
