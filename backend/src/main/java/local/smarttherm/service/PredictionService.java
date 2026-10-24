package local.smarttherm.service;

import local.smarttherm.model.PredictionForecast;
import local.smarttherm.model.PredictionSnapshot;
import local.smarttherm.model.ThermalReading;
import org.springframework.stereotype.Service;

import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.Optional;

@Service
public class PredictionService {
    public static final String MODEL = "Validated trained model prediction";
    private static final Duration HORIZON = Duration.ofMinutes(10);
    private static final Duration MAX_AGE = Duration.ofSeconds(15);
    private final PythonPredictionClient pythonPredictionClient;
    private volatile PredictionSnapshot latest = waitingForData();

    public PredictionService(PythonPredictionClient pythonPredictionClient) {
        this.pythonPredictionClient = pythonPredictionClient;
    }

    public PredictionSnapshot latest() {
        return latest;
    }

    public synchronized PredictionSnapshot predict(List<ThermalReading> readings) {
        Instant generatedAt = Instant.now();
        if (readings.isEmpty()) {
            latest = waitingForData();
            return latest;
        }
        ThermalReading newest = readings.getLast();
        if (!newest.sourceAvailable()
                || Duration.between(newest.timestamp(), generatedAt).abs().compareTo(MAX_AGE) > 0) {
            latest = new PredictionSnapshot(
                    "STALE_DATA", "Waiting for fresh sensor data", false, generatedAt, null, List.of());
            return latest;
        }

        PythonPredictionClient.InferenceResult inference = pythonPredictionClient.predict(readings);
        Optional<PredictionSnapshot> trained = inference.prediction();
        if (trained.isPresent()) {
            latest = trained.get();
            return latest;
        }

        latest = modelNotReady(newest, generatedAt, inference.reason());
        return latest;
    }

    public synchronized void markStale() {
        latest = new PredictionSnapshot(
                "STALE_DATA", "Waiting for fresh sensor data", false, Instant.now(), null, List.of());
    }

    private static PredictionSnapshot modelNotReady(
            ThermalReading newest,
            Instant generatedAt,
            String reason) {
        boolean insufficientHistory = reason != null
                && reason.toLowerCase().contains("history");
        String status = insufficientHistory ? "INSUFFICIENT_HISTORY" : "MODEL_NOT_READY";
        String forecastStatus = insufficientHistory ? "INSUFFICIENT_HISTORY" : "MODEL_UNAVAILABLE";
        String model = "Trained model not ready";
        if (reason != null && !reason.isBlank()) {
            model += ": " + reason;
        }
        Instant target = generatedAt.plus(HORIZON);
        List<PredictionForecast> forecasts = List.of(
                unavailableForecast("CPU", newest.cpuTemperatureC(), generatedAt, target, forecastStatus, model),
                unavailableForecast("GPU", newest.gpuTemperatureC(), generatedAt, target, forecastStatus, model));
        return new PredictionSnapshot(status, model, false, generatedAt, target, forecasts);
    }

    private static PredictionForecast unavailableForecast(
            String component,
            Double current,
            Instant generatedAt,
            Instant target,
            String status,
            String model) {
        return new PredictionForecast(component, current, null, generatedAt, target, status, model);
    }

    private static PredictionSnapshot waitingForData() {
        return new PredictionSnapshot(
                "WAITING_FOR_DATA", "Waiting for sensor data", false, null, null, List.of());
    }
}
