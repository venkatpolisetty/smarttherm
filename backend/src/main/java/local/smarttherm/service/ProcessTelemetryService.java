package local.smarttherm.service;

import local.smarttherm.model.ProcessUsage;
import local.smarttherm.model.ProcessUsage.ProcessEntry;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.web.server.ResponseStatusException;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.stream.Stream;

@Service
public class ProcessTelemetryService {
    private static final Duration FRESHNESS = Duration.ofSeconds(20);
    private static final Set<String> PROTECTED = Set.of("system", "registry", "smss", "csrss",
            "wininit", "services", "lsass", "svchost", "winlogon", "fontdrvhost");
    private final Path projectDirectory;

    public ProcessTelemetryService(@Value("${user.dir}") String projectDirectory) {
        this.projectDirectory = Path.of(projectDirectory).toAbsolutePath().normalize();
    }

    public ProcessUsage latest() {
        Path latest;
        try (Stream<Path> files = Files.list(projectDirectory)) {
            latest = files.filter(path -> path.getFileName().toString().startsWith("process_telemetry_")
                            && path.getFileName().toString().endsWith(".csv"))
                    .filter(Files::isRegularFile)
                    .max(Comparator.comparingLong(this::modifiedTime))
                    .orElse(null);
        } catch (IOException exception) {
            return unavailable("Could not read process telemetry.");
        }
        if (latest == null) return unavailable("Waiting for process CPU samples from the collector.");
        try {
            Instant newest = null;
            List<ProcessEntry> entries = new ArrayList<>();
            try (var reader = Files.newBufferedReader(latest)) {
                reader.readLine(); // CSV header
                String line;
                while ((line = reader.readLine()) != null) {
                    String[] columns = line.split(",", 7);
                    if (columns.length < 7) continue;
                    try {
                        Instant at = Instant.parse(columns[0]);
                        if (newest == null || at.isAfter(newest)) {
                            newest = at;
                            entries.clear();
                        }
                        if (!at.equals(newest)) continue;
                        double cpu = Double.parseDouble(columns[3]);
                        double memory = Double.parseDouble(columns[4]);
                        int pid = Integer.parseInt(columns[1]);
                        if (pid > 0 && Double.isFinite(cpu) && cpu >= 0 && cpu <= 100
                                && Double.isFinite(memory) && memory >= 0) {
                            entries.add(new ProcessEntry(pid, columns[2], cpu, memory,
                                    Boolean.parseBoolean(columns[5])));
                        }
                    } catch (RuntimeException ignored) {
                        // Ignore incomplete or invalid rows while the collector is appending.
                    }
                }
            }
            if (newest == null || Duration.between(newest, Instant.now()).abs().compareTo(FRESHNESS) > 0)
                return new ProcessUsage(newest, false, "Process CPU data is stale; waiting for the collector.", List.of());
            return new ProcessUsage(newest, true, "Live process usage", entries.stream()
                    .sorted(Comparator.comparingDouble(ProcessEntry::cpuUsagePercent).reversed()).toList());
        } catch (IOException exception) {
            return unavailable("Could not read process telemetry.");
        }
    }

    public void terminate(int pid) {
        if (pid <= 4 || pid == ProcessHandle.current().pid())
            throw new ResponseStatusException(HttpStatus.FORBIDDEN, "This process is protected.");
        ProcessEntry sampled = latest().processes().stream().filter(p -> p.processId() == pid).findFirst()
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.CONFLICT,
                        "Process is no longer in the latest CPU sample; refresh the list."));
        String name = sampled.processName().toLowerCase(Locale.ROOT);
        if (PROTECTED.contains(name) || name.equals("java") && isSmartTherm(pid))
            throw new ResponseStatusException(HttpStatus.FORBIDDEN, "This system or SmartTherm process is protected.");
        ProcessHandle handle = ProcessHandle.of(pid).orElseThrow(() ->
                new ResponseStatusException(HttpStatus.NOT_FOUND, "Process has already exited."));
        String actual = handle.info().command().map(Path::of).map(Path::getFileName).map(Path::toString)
                .orElse("").replaceFirst("(?i)\\.exe$", "");
        if (!actual.isBlank() && !actual.equalsIgnoreCase(sampled.processName()))
            throw new ResponseStatusException(HttpStatus.CONFLICT, "Process identity changed; refresh the list.");
        if (!handle.destroy()) throw new ResponseStatusException(HttpStatus.CONFLICT, "Windows did not allow the app to close.");
    }

    private boolean isSmartTherm(int pid) {
        return ProcessHandle.of(pid).flatMap(handle -> handle.info().commandLine())
                .map(command -> command.toLowerCase(Locale.ROOT).contains("smarttherm"))
                .orElse(false);
    }

    private long modifiedTime(Path path) {
        try { return Files.getLastModifiedTime(path).toMillis(); }
        catch (IOException ignored) { return 0; }
    }

    private ProcessUsage unavailable(String message) {
        return new ProcessUsage(null, false, message, List.of());
    }
}
