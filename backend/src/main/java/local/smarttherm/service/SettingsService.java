package local.smarttherm.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import local.smarttherm.model.AppSettings;
import jakarta.annotation.PostConstruct;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

@Service
public class SettingsService {
    private static final Logger logger = LoggerFactory.getLogger(SettingsService.class);
    private final ObjectMapper objectMapper;
    private final Path settingsFile;
    private AppSettings settings = AppSettings.defaults();

    public SettingsService(
            ObjectMapper objectMapper,
            @Value("${smarttherm.settings-file}") String settingsFile) {
        this.objectMapper = objectMapper;
        this.settingsFile = Path.of(settingsFile);
    }

    @PostConstruct
    synchronized void load() {
        if (!Files.isRegularFile(settingsFile)) {
            return;
        }
        try {
            AppSettings loaded = objectMapper.readValue(settingsFile.toFile(), AppSettings.class);
            if (valid(loaded)) {
                settings = loaded;
            } else {
                logger.warn("Ignoring invalid saved settings at {}", settingsFile);
            }
        } catch (IOException exception) {
            logger.warn("Could not load settings from {}: {}", settingsFile, exception.getMessage());
        }
    }

    public synchronized AppSettings get() {
        return settings;
    }

    public synchronized AppSettings update(AppSettings updated) {
        if (!valid(updated)) {
            throw new IllegalArgumentException("Warning thresholds must be between 50°C and 110°C.");
        }
        try {
            Files.createDirectories(settingsFile.getParent());
            Path temporary = settingsFile.resolveSibling(settingsFile.getFileName() + ".tmp");
            objectMapper.writeValue(temporary.toFile(), updated);
            Files.move(temporary, settingsFile, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        } catch (IOException exception) {
            throw new IllegalStateException("Could not save settings: " + exception.getMessage(), exception);
        }
        settings = updated;
        return settings;
    }

    private static boolean valid(AppSettings value) {
        return value != null
                && value.cpuWarningC() != null && Double.isFinite(value.cpuWarningC())
                && value.cpuWarningC() >= 50 && value.cpuWarningC() <= 110
                && value.gpuWarningC() != null && Double.isFinite(value.gpuWarningC())
                && value.gpuWarningC() >= 50 && value.gpuWarningC() <= 110
                && value.notificationsEnabled() != null;
    }
}
