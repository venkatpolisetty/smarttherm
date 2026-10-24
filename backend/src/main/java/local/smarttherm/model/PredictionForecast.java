package local.smarttherm.model;

import java.time.Instant;

public record PredictionForecast(
        String component,
        Double currentC,
        Double predictedC,
        Instant generatedAt,
        Instant targetTime,
        String status,
        String model) {
}
