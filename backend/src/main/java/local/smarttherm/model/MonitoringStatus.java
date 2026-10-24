package local.smarttherm.model;

import java.time.Instant;

public record MonitoringStatus(
        boolean active,
        boolean sourceConnected,
        Instant lastUpdate,
        String message) {
}
