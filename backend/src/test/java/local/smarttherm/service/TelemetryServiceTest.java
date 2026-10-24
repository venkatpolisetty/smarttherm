package local.smarttherm.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import local.smarttherm.model.ThermalReading;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;

import static org.assertj.core.api.Assertions.assertThat;

class TelemetryServiceTest {
    @TempDir
    Path temporaryDirectory;

    @Test
    void acceptsFreshValidatedSensorReadingsFromTheLocalFeed() throws Exception {
        ObjectMapper mapper = new ObjectMapper().registerModule(new JavaTimeModule());
        Path snapshot = temporaryDirectory.resolve("current-reading.json");
        Path history = temporaryDirectory.resolve("reading-history.jsonl");
        ThermalReading reading = new ThermalReading(
                Instant.now(), true, 61.5, 48.0, 23.0, 2_400.0,
                32.0, 1_200.0, 18.0, 42.0, "session-test");
        mapper.writeValue(snapshot.toFile(), reading);
        TelemetryService service = new TelemetryService(mapper, snapshot.toString(), history.toString());

        service.pollCurrentReading();

        assertThat(service.current()).isEqualTo(reading);
        assertThat(service.status().active()).isTrue();
        assertThat(service.recent(10)).containsExactly(reading);
    }

    @Test
    void rejectsOutOfRangeReadingsWithoutInventingAReplacement() throws Exception {
        ObjectMapper mapper = new ObjectMapper().registerModule(new JavaTimeModule());
        Path snapshot = temporaryDirectory.resolve("current-reading.json");
        ThermalReading invalid = new ThermalReading(
                Instant.now(), true, 250.0, null, null, null, null, null, null, null, "session-test");
        mapper.writeValue(snapshot.toFile(), invalid);
        TelemetryService service = new TelemetryService(
                mapper, snapshot.toString(), temporaryDirectory.resolve("history.jsonl").toString());

        service.pollCurrentReading();

        assertThat(service.current()).isNull();
        assertThat(service.status().active()).isFalse();
    }

    @Test
    void reportsMissingCollectorFeedAsDisconnected() {
        TelemetryService service = new TelemetryService(
                new ObjectMapper().registerModule(new JavaTimeModule()),
                temporaryDirectory.resolve("missing.json").toString(),
                temporaryDirectory.resolve("missing-history.jsonl").toString());

        service.pollCurrentReading();

        assertThat(service.current()).isNull();
        assertThat(service.status().message()).contains("managed collector");
    }

    @Test
    void keepsTheActiveInferenceWindowSeparateFromRetainedHistory() throws Exception {
        ObjectMapper mapper = new ObjectMapper().registerModule(new JavaTimeModule());
        Path history = temporaryDirectory.resolve("reading-history.jsonl");
        ThermalReading expired = reading(Instant.now().minusSeconds(900));
        ThermalReading recent = reading(Instant.now().minusSeconds(590));
        mapper.writeValue(history.toFile(), expired);
        Files.writeString(history, System.lineSeparator() + mapper.writeValueAsString(recent)
                + System.lineSeparator(), java.nio.file.StandardOpenOption.APPEND);
        TelemetryService service = new TelemetryService(
                mapper,
                temporaryDirectory.resolve("current-reading.json").toString(),
                history.toString());
        service.loadRecentHistory();

        assertThat(service.allRecent()).containsExactly(expired, recent);
        assertThat(service.activeWindow()).containsExactly(recent);
    }

    private static ThermalReading reading(Instant timestamp) {
        return new ThermalReading(
                timestamp, true, 61.5, null, 23.0, 2_400.0,
                32.0, 1_200.0, null, 42.0, "session-test");
    }
}
