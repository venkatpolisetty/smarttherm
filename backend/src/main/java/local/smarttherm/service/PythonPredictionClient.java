package local.smarttherm.service;

import local.smarttherm.model.PredictionForecast;
import local.smarttherm.model.PredictionSnapshot;
import local.smarttherm.model.ThermalReading;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientException;
import org.springframework.web.client.RestClientResponseException;

import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Optional;

@Service
public class PythonPredictionClient {
    private static final Logger logger = LoggerFactory.getLogger(PythonPredictionClient.class);
    private static final Duration LOOKBACK = Duration.ofMinutes(10);
    private static final int MAX_READINGS = 150;
    private static final Duration TARGET_HORIZON = Duration.ofMinutes(10);
    private static final Duration TARGET_TOLERANCE = Duration.ofSeconds(30);
    private static final Duration ERROR_LOG_INTERVAL = Duration.ofMinutes(1);

    private final RestClient restClient;
    private final boolean enabled;
    private String lastLoggedIssue;
    private Instant lastIssueLog;

    public PythonPredictionClient(
            RestClient smartThermMlRestClient,
            @Value("${smarttherm.ml.enabled:true}") boolean enabled) {
        this.restClient = smartThermMlRestClient;
        this.enabled = enabled;
    }

    public InferenceResult predict(List<ThermalReading> readings) {
        if (!enabled) {
            return unavailable("Local trained-model inference is disabled.");
        }
        Instant cutoff = Instant.now().minus(LOOKBACK);
        List<ThermalReading> requestReadings = readings.stream()
                .filter(reading -> reading.timestamp() != null)
                .filter(reading -> !reading.timestamp().isBefore(cutoff))
                .toList();
        if (requestReadings.size() > MAX_READINGS) {
            requestReadings = requestReadings.subList(
                    requestReadings.size() - MAX_READINGS, requestReadings.size());
        }
        if (requestReadings.isEmpty()) {
            return unavailable("No recent sensor readings are available for model inference.");
        }
        try {
            PredictionSnapshot response = restClient.post()
                    .uri("/predict")
                    .body(new PredictionRequest(requestReadings))
                    .retrieve()
                    .body(PredictionSnapshot.class);
            if (!validTrainedResponse(response, requestReadings.getLast())) {
                String reason = response == null
                        ? "Python inference returned an empty response."
                        : response.status().equals("MODEL_UNAVAILABLE")
                        ? "No validated trained model is installed."
                        : response.status().equals("INSUFFICIENT_HISTORY")
                        ? "The trained model is ready, but recent sensor history is insufficient."
                        : "Python inference did not return a valid trained-model prediction.";
                logIssue(reason, null);
                return new InferenceResult(Optional.empty(), reason);
            }
            lastLoggedIssue = null;
            lastIssueLog = null;
            return new InferenceResult(Optional.of(response), "");
        } catch (RestClientResponseException exception) {
            String reason = "Local Python inference request failed with HTTP "
                    + exception.getStatusCode().value() + "; no prediction is available.";
            logIssue(reason, exception);
            return new InferenceResult(Optional.empty(), reason);
        } catch (RestClientException exception) {
            String reason = "Local Python inference service is unavailable; no prediction is available.";
            logIssue(reason, exception);
            return new InferenceResult(Optional.empty(), reason);
        }
    }

    public boolean requestTrainingCheck() {
        if (!enabled) {
            return false;
        }
        try {
            restClient.post()
                    .uri("/training/check")
                    .retrieve()
                    .body(Map.class);
            return true;
        } catch (RestClientException exception) {
            logger.warn("Could not request an immediate local model training check: {}", exception.getMessage());
            return false;
        }
    }

    private static boolean validTrainedResponse(
            PredictionSnapshot response,
            ThermalReading latestReading) {
        if (response == null || !response.trainedModel()
                || !"READY".equals(response.status())
                || response.model() == null || response.model().isBlank()
                || response.generatedAt() == null || response.targetTime() == null
                || response.forecasts() == null) {
            return false;
        }
        Instant now = Instant.now();
        Instant expectedTarget = response.generatedAt().plus(TARGET_HORIZON);
        if (Duration.between(expectedTarget, response.targetTime()).abs()
                .compareTo(TARGET_TOLERANCE) > 0
                || Duration.between(now, response.generatedAt()).abs()
                .compareTo(TARGET_TOLERANCE) > 0) {
            return false;
        }
        if (response.forecasts().size() != 2) {
            return false;
        }
        boolean cpu = false;
        boolean gpu = false;
        boolean ready = false;
        for (PredictionForecast forecast : response.forecasts()) {
            if (forecast == null
                    || forecast.generatedAt() == null
                    || forecast.targetTime() == null
                    || Duration.between(response.generatedAt(), forecast.generatedAt()).abs()
                        .compareTo(TARGET_TOLERANCE) > 0
                    || Duration.between(response.targetTime(), forecast.targetTime()).abs()
                        .compareTo(TARGET_TOLERANCE) > 0) {
                return false;
            }
            if ("CPU".equals(forecast.component()) && !cpu
                    && matches(forecast.currentC(), latestReading.cpuTemperatureC())) {
                cpu = true;
            } else if ("GPU".equals(forecast.component()) && !gpu
                    && matches(forecast.currentC(), latestReading.gpuTemperatureC())) {
                gpu = true;
            } else {
                return false;
            }
            if ("READY".equals(forecast.status())) {
                if (!validForecast(forecast)) {
                    return false;
                }
                ready = true;
            } else if (!List.of("MODEL_UNAVAILABLE", "INSUFFICIENT_HISTORY", "MODEL_ERROR")
                    .contains(forecast.status())
                    || forecast.predictedC() != null) {
                return false;
            }
        }
        return cpu && gpu && ready;
    }

    private static boolean validForecast(PredictionForecast forecast) {
        return forecast != null
                && ("CPU".equals(forecast.component()) || "GPU".equals(forecast.component()))
                && "READY".equals(forecast.status())
                && forecast.model() != null && !forecast.model().isBlank()
                && forecast.currentC() != null
                && Double.isFinite(forecast.currentC())
                && forecast.currentC() >= 0 && forecast.currentC() <= 150
                && forecast.predictedC() != null
                && Double.isFinite(forecast.predictedC())
                && forecast.predictedC() >= 0 && forecast.predictedC() <= 150;
    }

    private static boolean matches(Double actual, Double expected) {
        if (actual == null || expected == null) {
            return actual == expected;
        }
        return Double.isFinite(actual) && Math.abs(actual - expected) <= 0.1;
    }

    private synchronized void logIssue(String reason, Exception exception) {
        Instant now = Instant.now();
        if (!reason.equals(lastLoggedIssue)
                || lastIssueLog == null
                || Duration.between(lastIssueLog, now).compareTo(ERROR_LOG_INTERVAL) >= 0) {
            if (exception == null) {
                logger.warn("{}", reason);
            } else {
                logger.warn("{} ({})", reason, exception.getMessage());
            }
            lastLoggedIssue = reason;
            lastIssueLog = now;
        }
    }

    private InferenceResult unavailable(String reason) {
        logIssue(reason, null);
        return new InferenceResult(Optional.empty(), reason);
    }

    public record InferenceResult(Optional<PredictionSnapshot> prediction, String reason) {
    }

    public record PredictionRequest(List<ThermalReading> readings) {
    }
}
