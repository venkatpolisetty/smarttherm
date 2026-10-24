package local.smarttherm.service;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import local.smarttherm.model.AppSettings;
import local.smarttherm.model.PredictionForecast;
import local.smarttherm.model.PredictionSnapshot;
import local.smarttherm.model.ThermalAlert;
import jakarta.annotation.PostConstruct;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;

@Service
public class AlertService {
    private static final Logger logger = LoggerFactory.getLogger(AlertService.class);
    private static final int MAX_ALERTS = 100;
    private static final Duration COOLDOWN = Duration.ofMinutes(10);

    private final ObjectMapper objectMapper;
    private final Path alertsFile;
    private final List<ThermalAlert> alerts = new ArrayList<>();
    private final Map<String, Instant> lastCreated = new HashMap<>();
    private final Map<String, Boolean> activeByComponent = new HashMap<>();

    public AlertService(
            ObjectMapper objectMapper,
            @Value("${smarttherm.alerts-file:${user.dir}/backend/data/alerts.json}") String alertsFile) {
        this.objectMapper = objectMapper;
        this.alertsFile = Path.of(alertsFile);
    }

    @PostConstruct
    void load() {
        if (!Files.isRegularFile(alertsFile)) {
            return;
        }
        try {
            List<ThermalAlert> saved = objectMapper.readValue(
                    alertsFile.toFile(), new TypeReference<>() {});
            synchronized (this) {
                List<ThermalAlert> newestFirst = saved.stream()
                        .sorted(Comparator.comparing(ThermalAlert::createdAt).reversed())
                        .limit(MAX_ALERTS)
                        .toList();
                alerts.addAll(newestFirst);
                Set<String> seenComponents = new HashSet<>();
                for (ThermalAlert alert : newestFirst) {
                    if (!seenComponents.add(alert.component())) {
                        continue;
                    }
                    activeByComponent.put(alert.component(), alert.active());
                    lastCreated.put(alert.component(), alert.createdAt());
                }
            }
        } catch (IOException exception) {
            logger.warn("Could not load alert history from {}: {}", alertsFile, exception.getMessage());
        }
    }

    public synchronized List<ThermalAlert> recent() {
        return List.copyOf(alerts);
    }

    public synchronized void evaluate(PredictionSnapshot prediction, AppSettings settings) {
        if (!prediction.trainedModel() || prediction.forecasts() == null) {
            return;
        }
        for (PredictionForecast forecast : prediction.forecasts()) {
            if (!"READY".equals(forecast.status())) {
                continue;
            }
            Double threshold = "CPU".equals(forecast.component())
                    ? settings.cpuWarningC() : settings.gpuWarningC();
            if (forecast.predictedC() == null) {
                continue;
            }
            boolean crossing = forecast.predictedC() >= threshold;
            if (!crossing) {
                if (forecast.predictedC() <= threshold - 2) {
                    resolve(forecast.component());
                }
                continue;
            }
            if (Boolean.TRUE.equals(activeByComponent.get(forecast.component()))) {
                continue;
            }
            Instant now = Instant.now();
            Instant previous = lastCreated.get(forecast.component());
            if (previous != null && Duration.between(previous, now).compareTo(COOLDOWN) < 0) {
                continue;
            }
            String severity = forecast.predictedC() >= threshold + 5 ? "CRITICAL" : "WARNING";
            String name = forecast.component();
            ThermalAlert alert = new ThermalAlert(
                    UUID.randomUUID().toString(),
                    now,
                    name,
                    forecast.currentC(),
                    forecast.predictedC(),
                    threshold,
                    severity,
                    String.format("%s temperature is predicted to reach %.1f°C within approximately 10 minutes.",
                            name, forecast.predictedC()),
                    "Reduce demanding workloads and check that the computer's vents are unobstructed.",
                    true);
            alerts.add(0, alert);
            if (alerts.size() > MAX_ALERTS) {
                alerts.remove(alerts.size() - 1);
            }
            activeByComponent.put(name, true);
            lastCreated.put(name, now);
            persist();
        }
    }

    private void resolve(String component) {
        if (!Boolean.TRUE.equals(activeByComponent.get(component))) {
            return;
        }
        activeByComponent.put(component, false);
        for (int index = 0; index < alerts.size(); index++) {
            ThermalAlert alert = alerts.get(index);
            if (alert.component().equals(component) && alert.active()) {
                alerts.set(index, new ThermalAlert(
                        alert.id(), alert.createdAt(), alert.component(), alert.currentC(),
                        alert.predictedC(), alert.thresholdC(), alert.severity(), alert.message(),
                        alert.suggestedAction(), false));
                persist();
                return;
            }
        }
    }

    private void persist() {
        try {
            Files.createDirectories(alertsFile.getParent());
            Path temporary = alertsFile.resolveSibling(alertsFile.getFileName() + ".tmp");
            objectMapper.writeValue(temporary.toFile(), alerts);
            Files.move(temporary, alertsFile, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        } catch (IOException exception) {
            logger.error("Could not persist alert history: {}", exception.getMessage());
        }
    }
}
