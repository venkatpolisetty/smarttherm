package local.smarttherm.api;

import jakarta.validation.Valid;
import local.smarttherm.model.AppSettings;
import local.smarttherm.model.MonitoringStatus;
import local.smarttherm.model.PredictionSnapshot;
import local.smarttherm.model.ThermalAlert;
import local.smarttherm.model.ThermalReading;
import local.smarttherm.model.ProcessUsage;
import local.smarttherm.service.AlertService;
import local.smarttherm.service.PredictionService;
import local.smarttherm.service.SettingsService;
import local.smarttherm.service.TelemetryService;
import local.smarttherm.service.ProcessTelemetryService;
import local.smarttherm.service.CollectorProcessManager;
import local.smarttherm.service.PythonPredictionClient;
import org.springframework.http.ResponseEntity;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ResponseStatusException;

import java.time.Instant;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api")
public class ApiController {
    private final TelemetryService telemetryService;
    private final PredictionService predictionService;
    private final SettingsService settingsService;
    private final AlertService alertService;
    private final ProcessTelemetryService processTelemetryService;
    private final CollectorProcessManager collectorProcessManager;
    private final PythonPredictionClient pythonPredictionClient;

    public ApiController(
            TelemetryService telemetryService,
            PredictionService predictionService,
            SettingsService settingsService,
            AlertService alertService,
            ProcessTelemetryService processTelemetryService,
            CollectorProcessManager collectorProcessManager,
            PythonPredictionClient pythonPredictionClient) {
        this.telemetryService = telemetryService;
        this.predictionService = predictionService;
        this.settingsService = settingsService;
        this.alertService = alertService;
        this.processTelemetryService = processTelemetryService;
        this.collectorProcessManager = collectorProcessManager;
        this.pythonPredictionClient = pythonPredictionClient;
    }

    @GetMapping("/health")
    public Map<String, Object> health() {
        return Map.of("status", "UP", "time", Instant.now());
    }

    @PostMapping("/calibration/start")
    public Map<String, Object> startCalibration() {
        boolean collectorStarted = collectorProcessManager.startCollection();
        boolean trainingCheckRequested = pythonPredictionClient.requestTrainingCheck();
        String message = collectorStarted
                ? "Calibration started. SmartTherm is collecting real readings; model training and updates happen automatically when enough valid data is available."
                : "Calibration could not start the sensor collector. Check that collection is enabled and the .NET collector can run.";
        return Map.of(
                "collectorStarted", collectorStarted,
                "trainingCheckRequested", trainingCheckRequested,
                "message", message);
    }

    @GetMapping("/calibration/status")
    public Map<String, Object> calibrationStatus() {
        return Map.of("collecting", collectorProcessManager.isCollectionRunning());
    }

    @PostMapping("/calibration/stop")
    public Map<String, Object> stopCalibration() {
        boolean collectorStopped = collectorProcessManager.stopCollection();
        return Map.of(
                "collectorStopped", collectorStopped,
                "message", collectorStopped
                        ? "Calibration stopped. Sensor collection is paused; live readings will become unavailable."
                        : "The sensor collector did not stop cleanly. Check the backend logs.");
    }

    @GetMapping("/monitoring/status")
    public MonitoringStatus status() {
        return telemetryService.status();
    }

    @GetMapping("/temperature/current")
    public ResponseEntity<ThermalReading> current() {
        ThermalReading reading = telemetryService.current();
        return reading == null ? ResponseEntity.noContent().build() : ResponseEntity.ok(reading);
    }

    @GetMapping("/temperature/history")
    public List<ThermalReading> history(@RequestParam(defaultValue = "600") int limit) {
        return telemetryService.recent(Math.max(1, Math.min(limit, 2_000)));
    }

    @GetMapping("/predictions/latest")
    public PredictionSnapshot prediction() {
        return predictionService.latest();
    }

    @GetMapping("/alerts")
    public List<ThermalAlert> alerts() {
        return alertService.recent();
    }

    @GetMapping("/settings")
    public AppSettings settings() {
        return settingsService.get();
    }

    @GetMapping("/processes")
    public ProcessUsage processes() {
        return processTelemetryService.latest();
    }

    @PostMapping("/processes/{pid}/terminate")
    public ResponseEntity<Void> terminateProcess(@PathVariable int pid) {
        processTelemetryService.terminate(pid);
        return ResponseEntity.noContent().build();
    }

    @PutMapping("/settings")
    public AppSettings updateSettings(@Valid @RequestBody AppSettings settings) {
        try {
            return settingsService.update(settings);
        } catch (IllegalArgumentException exception) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, exception.getMessage(), exception);
        } catch (IllegalStateException exception) {
            throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR,
                    exception.getMessage(), exception);
        }
    }
}
