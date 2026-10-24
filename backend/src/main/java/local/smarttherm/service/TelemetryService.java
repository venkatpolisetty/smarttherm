package local.smarttherm.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import local.smarttherm.model.MonitoringStatus;
import local.smarttherm.model.ThermalReading;
import jakarta.annotation.PostConstruct;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;

@Service
public class TelemetryService {
    private static final Logger logger = LoggerFactory.getLogger(TelemetryService.class);
    private static final int MAX_HISTORY = 2_000;
    private static final int MAX_ACTIVE_READINGS = 150;
    private static final Duration ACTIVE_WINDOW = Duration.ofSeconds(600);
    private static final Duration FRESHNESS = Duration.ofSeconds(15);

    private final ObjectMapper objectMapper;
    private final Path currentFile;
    private final Path historyFile;
    private final ArrayDeque<ThermalReading> history = new ArrayDeque<>();
    private final ArrayDeque<ThermalReading> activeWindow = new ArrayDeque<>();
    private Instant lastSeen;
    private String lastError = "Waiting for the hardware collector.";

    public TelemetryService(
            ObjectMapper objectMapper,
            @Value("${smarttherm.sensor-file}") String currentFile,
            @Value("${smarttherm.sensor-history-file}") String historyFile) {
        this.objectMapper = objectMapper;
        this.currentFile = Path.of(currentFile);
        this.historyFile = Path.of(historyFile);
    }

    @PostConstruct
    void loadRecentHistory() {
        if (!Files.isRegularFile(historyFile)) {
            return;
        }
        try (var lines = Files.lines(historyFile)) {
            lines.forEach(line -> {
                try {
                    addIfValid(objectMapper.readValue(line, ThermalReading.class));
                } catch (IOException exception) {
                    logger.warn("Ignoring an invalid line in sensor history: {}", exception.getMessage());
                }
            });
        } catch (IOException exception) {
            logger.warn("Could not load sensor history from {}: {}", historyFile, exception.getMessage());
        }
    }

    @Scheduled(fixedDelayString = "${smarttherm.poll-ms:1000}")
    public void pollCurrentReading() {
        synchronized (this) {
            pruneActiveWindow(Instant.now());
        }
        if (!Files.isRegularFile(currentFile)) {
            setError("Hardware feed is unavailable. Check the managed collector startup logs.");
            return;
        }
        try {
            ThermalReading reading = objectMapper.readValue(currentFile.toFile(), ThermalReading.class);
            if (!isValid(reading)) {
                setError("Hardware feed contains no valid sensor metrics.");
                return;
            }
            synchronized (this) {
                if (lastSeen == null || reading.timestamp().isAfter(lastSeen)) {
                    addIfValid(reading);
                    lastSeen = reading.timestamp();
                    lastError = null;
                }
            }
        } catch (IOException exception) {
            setError("Could not read hardware feed: " + exception.getMessage());
        }
    }

    public synchronized ThermalReading current() {
        return history.peekLast();
    }

    public synchronized List<ThermalReading> recent(int limit) {
        int boundedLimit = Math.max(1, Math.min(limit, MAX_HISTORY));
        List<ThermalReading> readings = new ArrayList<>(history);
        return readings.subList(Math.max(0, readings.size() - boundedLimit), readings.size());
    }

    public synchronized List<ThermalReading> allRecent() {
        return List.copyOf(history);
    }

    public synchronized List<ThermalReading> activeWindow() {
        pruneActiveWindow(Instant.now());
        return List.copyOf(activeWindow);
    }

    public synchronized MonitoringStatus status() {
        ThermalReading reading = history.peekLast();
        boolean fresh = reading != null
                && reading.sourceAvailable()
                && Duration.between(reading.timestamp(), Instant.now()).abs().compareTo(FRESHNESS) <= 0;
        String message = fresh
                ? "Connected to the local hardware sensor collector."
                : (lastError != null ? lastError : "Hardware feed is stale or disconnected.");
        return new MonitoringStatus(fresh, fresh, reading == null ? null : reading.timestamp(), message);
    }

    private void setError(String error) {
        synchronized (this) {
            if (!error.equals(lastError)) {
                logger.warn("{}", error);
                lastError = error;
            }
        }
    }

    private synchronized void addIfValid(ThermalReading reading) {
        if (!isValid(reading) || (lastSeen != null && !reading.timestamp().isAfter(lastSeen))) {
            return;
        }
        history.addLast(reading);
        while (history.size() > MAX_HISTORY) {
            history.removeFirst();
        }
        activeWindow.addLast(reading);
        pruneActiveWindow(Instant.now());
        while (activeWindow.size() > MAX_ACTIVE_READINGS) {
            activeWindow.removeFirst();
        }
        lastSeen = reading.timestamp();
    }

    private void pruneActiveWindow(Instant now) {
        Instant cutoff = now.minus(ACTIVE_WINDOW);
        while (!activeWindow.isEmpty()
                && activeWindow.peekFirst().timestamp().isBefore(cutoff)) {
            activeWindow.removeFirst();
        }
    }

    private static boolean isValid(ThermalReading reading) {
        return reading != null
                && reading.timestamp() != null
                && (validTemperature(reading.cpuTemperatureC())
                    || validTemperature(reading.gpuTemperatureC())
                    || validPercent(reading.cpuUtilizationPercent())
                    || validClock(reading.cpuClockMhz())
                    || validPower(reading.cpuPowerW())
                    || validFanSpeed(reading.cpuFanSpeedRpm())
                    || validPercent(reading.gpuUtilizationPercent())
                    || validPercent(reading.memoryUtilizationPercent()));
    }

    private static boolean validTemperature(Double value) {
        return value != null && Double.isFinite(value) && value >= 0 && value <= 150;
    }

    private static boolean validPercent(Double value) {
        return value != null && Double.isFinite(value) && value >= 0 && value <= 100;
    }

    private static boolean validClock(Double value) {
        return value != null && Double.isFinite(value) && value > 0 && value <= 10_000;
    }

    private static boolean validPower(Double value) {
        return value != null && Double.isFinite(value) && value >= 0 && value <= 1_000;
    }

    private static boolean validFanSpeed(Double value) {
        return value != null && Double.isFinite(value) && value >= 0 && value <= 100_000;
    }
}
