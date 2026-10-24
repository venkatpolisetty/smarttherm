package local.smarttherm.model;

import java.time.Instant;

public record ThermalAlert(
        String id,
        Instant createdAt,
        String component,
        Double currentC,
        Double predictedC,
        Double thresholdC,
        String severity,
        String message,
        String suggestedAction,
        boolean active) {
}
