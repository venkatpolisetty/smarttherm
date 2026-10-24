package local.smarttherm.service;

import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.time.Instant;

@Component
public class MonitoringCoordinator {
    private final TelemetryService telemetryService;
    private final PredictionService predictionService;
    private final SettingsService settingsService;
    private final AlertService alertService;
    private Instant lastProcessed;

    public MonitoringCoordinator(
            TelemetryService telemetryService,
            PredictionService predictionService,
            SettingsService settingsService,
            AlertService alertService) {
        this.telemetryService = telemetryService;
        this.predictionService = predictionService;
        this.settingsService = settingsService;
        this.alertService = alertService;
    }

    @Scheduled(fixedDelay = 1000)
    public synchronized void refreshPredictionAndAlerts() {
        var readings = telemetryService.allRecent();
        var current = readings.isEmpty() ? null : readings.getLast();
        if (current == null) {
            return;
        }
        if (current.timestamp().equals(lastProcessed)) {
            if (Instant.now().isAfter(current.timestamp().plusSeconds(15))) {
                predictionService.markStale();
            }
            return;
        }
        var prediction = predictionService.predict(telemetryService.activeWindow());
        alertService.evaluate(prediction, settingsService.get());
        lastProcessed = current.timestamp();
    }
}
