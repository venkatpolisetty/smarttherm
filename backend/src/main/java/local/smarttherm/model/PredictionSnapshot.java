package local.smarttherm.model;

import java.time.Instant;
import java.util.List;

public record PredictionSnapshot(
        String status,
        String model,
        boolean trainedModel,
        Instant generatedAt,
        Instant targetTime,
        List<PredictionForecast> forecasts) {
}
