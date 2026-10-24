package local.smarttherm.service;

import jakarta.annotation.PreDestroy;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Component;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

@Component
public class CollectorProcessManager {
    private static final Logger logger = LoggerFactory.getLogger(CollectorProcessManager.class);
    private static final long RESTART_DELAY_MILLIS = 5_000;

    private final Path projectFile;
    private final boolean enabled;
    private final AtomicBoolean stopping = new AtomicBoolean();
    private volatile Process process;
    private Thread supervisor;
    private boolean restartPaused;

    public CollectorProcessManager(
            @Value("${smarttherm.collector.project:${user.dir}/ThermalCollector/ThermalCollector.csproj}")
            String projectFile,
            @Value("${smarttherm.collector.enabled:true}") boolean enabled) {
        this.projectFile = Path.of(projectFile).toAbsolutePath().normalize();
        this.enabled = enabled;
    }

    @EventListener(ApplicationReadyEvent.class)
    public synchronized void start() {
        startSupervisor();
    }

    public synchronized boolean startCollection() {
        if (!enabled || stopping.get()) {
            return false;
        }
        restartPaused = false;
        if (process != null && !process.isAlive()) {
            process = null;
        }
        ensureStarted();
        startSupervisor();
        return process != null && process.isAlive();
    }

    public synchronized boolean stopCollection() {
        restartPaused = true;
        Process running = process;
        if (running == null || !running.isAlive()) {
            process = null;
            return true;
        }
        stopProcess(running);
        process = null;
        return !running.isAlive();
    }

    public synchronized boolean isCollectionRunning() {
        return process != null && process.isAlive() && !restartPaused;
    }

    private void startSupervisor() {
        if (!enabled) {
            logger.info("Managed sensor collection is disabled by configuration.");
            return;
        }
        if (supervisor == null || !supervisor.isAlive()) {
            supervisor = Thread.ofVirtual().start(this::supervise);
        }
    }

    @PreDestroy
    public synchronized void stop() {
        stopping.set(true);
        Process running = process;
        if (running == null || !running.isAlive()) {
            return;
        }
        stopProcess(running);
    }

    private void stopProcess(Process running) {
        try {
            try (BufferedWriter input = new BufferedWriter(
                    new OutputStreamWriter(running.getOutputStream(), StandardCharsets.UTF_8))) {
                input.write("stop");
                input.newLine();
                input.flush();
            }
            if (!running.waitFor(10, TimeUnit.SECONDS)) {
                logger.warn("Sensor collector did not stop cleanly; requesting process termination.");
                destroyProcessTree(running, false);
                if (!running.waitFor(3, TimeUnit.SECONDS)) {
                    destroyProcessTree(running, true);
                }
            }
        } catch (IOException exception) {
            logger.warn("Could not send a clean shutdown request to the sensor collector: {}",
                    exception.getMessage());
            destroyProcessTree(running, true);
        } catch (InterruptedException exception) {
            Thread.currentThread().interrupt();
            destroyProcessTree(running, true);
        }
    }

    private static void destroyProcessTree(Process process, boolean forcibly) {
        List<ProcessHandle> descendants = process.toHandle().descendants().toList();
        for (int index = descendants.size() - 1; index >= 0; index--) {
            ProcessHandle descendant = descendants.get(index);
            if (forcibly) {
                descendant.destroyForcibly();
            } else {
                descendant.destroy();
            }
        }
        if (forcibly) {
            process.destroyForcibly();
        } else {
            process.destroy();
        }
    }

    private void supervise() {
        while (!stopping.get()) {
            ensureStarted();
            try {
                Thread.sleep(RESTART_DELAY_MILLIS);
            } catch (InterruptedException exception) {
                Thread.currentThread().interrupt();
                return;
            }
        }
    }

    private synchronized void ensureStarted() {
        if (stopping.get() || restartPaused || (process != null && process.isAlive())) {
            return;
        }
        if (process != null) {
            int exitCode = process.exitValue();
            process = null;
            if (exitCode == 0) {
                restartPaused = true;
                logger.warn("Sensor collector exited normally; automatic restart is paused.");
                return;
            }
            logger.warn("Sensor collector exited with code {}; it will be restarted.", exitCode);
        }
        if (!Files.isRegularFile(projectFile)) {
            logger.error("Sensor collector project was not found at {}.", projectFile);
            return;
        }
        try {
            Process started = new ProcessBuilder(
                    "dotnet", "run", "--project", projectFile.toString())
                    .directory(projectFile.getParent().getParent().toFile())
                    .redirectErrorStream(true)
                    .start();
            process = started;
            logger.info("Started sensor collector from {}.", projectFile);
            Thread.ofVirtual().start(() -> logCollectorOutput(started));
        } catch (IOException exception) {
            logger.error("Could not start the .NET sensor collector: {}", exception.getMessage());
        }
    }

    private void logCollectorOutput(Process collector) {
        try (BufferedReader output = new BufferedReader(
                new InputStreamReader(collector.getInputStream(), StandardCharsets.UTF_8))) {
            String line;
            while ((line = output.readLine()) != null) {
                if (line.contains("warning") || line.contains("Warning") || line.contains("error")) {
                    logger.warn("Sensor collector: {}", line);
                } else {
                    logger.debug("Sensor collector: {}", line);
                }
            }
        } catch (IOException exception) {
            if (!stopping.get()) {
                logger.warn("Could not read sensor collector output: {}", exception.getMessage());
            }
        }
    }
}
