package local.smarttherm.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import local.smarttherm.model.AppSettings;
import local.smarttherm.model.PredictionForecast;
import local.smarttherm.model.PredictionSnapshot;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.time.Instant;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

class AlertServiceTest {
    @TempDir
    Path temporaryDirectory;

    @Test
    void emitsOneAlertForAnOngoingThresholdCrossing() {
        AlertService service = new AlertService(
                new ObjectMapper().registerModule(new JavaTimeModule()),
                temporaryDirectory.resolve("alerts.json").toString());
        PredictionForecast forecast = new PredictionForecast(
                "CPU", 80.0, 92.0, Instant.now(), Instant.now().plusSeconds(600),
                "READY", PredictionService.MODEL);
        PredictionSnapshot prediction = new PredictionSnapshot(
                "READY", PredictionService.MODEL, true, Instant.now(),
                Instant.now().plusSeconds(600), List.of(forecast));

        service.evaluate(prediction, AppSettings.defaults());
        service.evaluate(prediction, AppSettings.defaults());

        assertThat(service.recent()).hasSize(1);
        assertThat(service.recent().getFirst().message()).contains("CPU temperature");
        assertThat(service.recent().getFirst().active()).isTrue();
        assertThat(temporaryDirectory.resolve("alerts.json")).exists();
    }

    @Test
    void resetsAnAlertOnlyAfterTheEstimateFallsBelowTheHysteresisBand() {
        AlertService service = new AlertService(
                new ObjectMapper().registerModule(new JavaTimeModule()),
                temporaryDirectory.resolve("alerts.json").toString());
        service.evaluate(prediction(92.0), AppSettings.defaults());
        service.evaluate(prediction(87.0), AppSettings.defaults());

        assertThat(service.recent().getFirst().active()).isFalse();
    }

    @Test
    void cooldownPreventsAnImmediateRepeatAfterTheAlertResets() {
        AlertService service = new AlertService(
                new ObjectMapper().registerModule(new JavaTimeModule()),
                temporaryDirectory.resolve("alerts.json").toString());
        service.evaluate(prediction(92.0), AppSettings.defaults());
        service.evaluate(prediction(87.0), AppSettings.defaults());
        service.evaluate(prediction(92.0), AppSettings.defaults());

        assertThat(service.recent()).hasSize(1);
        assertThat(service.recent().getFirst().active()).isFalse();
    }

    @Test
    void doesNotCreateRiskAlertsFromUntrainedPredictions() {
        AlertService service = new AlertService(
                new ObjectMapper().registerModule(new JavaTimeModule()),
                temporaryDirectory.resolve("alerts.json").toString());

        service.evaluate(new PredictionSnapshot(
                "MODEL_NOT_READY", "No model", false, Instant.now(),
                Instant.now().plusSeconds(600),
                List.of(new PredictionForecast(
                        "CPU", 80.0, 120.0, Instant.now(), Instant.now().plusSeconds(600),
                        "READY", PredictionService.MODEL))), AppSettings.defaults());

        assertThat(service.recent()).isEmpty();
    }

    private static PredictionSnapshot prediction(double cpuC) {
        return new PredictionSnapshot("READY", PredictionService.MODEL, true,
                Instant.now(), Instant.now().plusSeconds(600),
                List.of(new PredictionForecast(
                        "CPU", 80.0, cpuC, Instant.now(), Instant.now().plusSeconds(600),
                        "READY", PredictionService.MODEL)));
    }
}
