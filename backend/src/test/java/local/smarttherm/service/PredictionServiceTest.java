package local.smarttherm.service;

import local.smarttherm.model.PredictionSnapshot;
import local.smarttherm.model.ThermalReading;
import org.junit.jupiter.api.Test;

import java.time.Instant;
import java.util.List;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class PredictionServiceTest {
    @Test
    void doesNotInventTemperatureWhenNoTrainedModelIsAvailable() {
        Instant now = Instant.now();
        List<ThermalReading> readings = List.of(
                reading(now.minusSeconds(20), 60.0),
                reading(now.minusSeconds(10), 62.0),
                reading(now, 64.0));
        PythonPredictionClient client = mock(PythonPredictionClient.class);
        when(client.predict(anyList())).thenReturn(
                new PythonPredictionClient.InferenceResult(
                        Optional.empty(), "No validated trained model is installed."));

        PredictionSnapshot result = new PredictionService(client).predict(readings);

        assertThat(result.status()).isEqualTo("MODEL_NOT_READY");
        assertThat(result.trainedModel()).isFalse();
        assertThat(result.model()).contains("No validated trained model");
        assertThat(result.targetTime()).isEqualTo(result.generatedAt().plusSeconds(300));
        assertThat(result.forecasts().getFirst().predictedC()).isNull();
    }

    @Test
    void reportsInsufficientHistoryWithoutMakingUpAPrediction() {
        PythonPredictionClient client = mock(PythonPredictionClient.class);
        when(client.predict(anyList())).thenReturn(
                new PythonPredictionClient.InferenceResult(
                        Optional.empty(), "Recent sensor history is insufficient."));
        PredictionSnapshot result = new PredictionService(client).predict(
                List.of(reading(Instant.now(), 60.0)));

        assertThat(result.status()).isEqualTo("INSUFFICIENT_HISTORY");
        assertThat(result.forecasts().getFirst().predictedC()).isNull();
    }

    @Test
    void passesTheFullRollingWindowToInference() {
        Instant now = Instant.now();
        List<ThermalReading> readings = List.of(
                reading(now.minusSeconds(600), 60.0),
                reading(now.minusSeconds(300), 62.0),
                reading(now, 64.0));
        PythonPredictionClient client = mock(PythonPredictionClient.class);
        when(client.predict(anyList())).thenReturn(
                new PythonPredictionClient.InferenceResult(
                        Optional.empty(), "No validated trained model is installed."));

        new PredictionService(client).predict(readings);

        verify(client).predict(readings);
    }

    @Test
    void prefersAValidatedTrainedModelPrediction() {
        Instant now = Instant.now();
        PredictionSnapshot trained = new PredictionSnapshot(
                "READY",
                "Python trained models: XGBoost",
                true,
                now,
                now.plusSeconds(300),
                List.of(new local.smarttherm.model.PredictionForecast(
                        "CPU", 64.0, 70.0, now, now.plusSeconds(300), "READY", "XGBoost")));
        PythonPredictionClient client = mock(PythonPredictionClient.class);
        when(client.predict(anyList())).thenReturn(
                new PythonPredictionClient.InferenceResult(Optional.of(trained), ""));

        PredictionSnapshot result = new PredictionService(client).predict(
                List.of(reading(now, 64.0)));

        assertThat(result.trainedModel()).isTrue();
        assertThat(result.model()).contains("XGBoost");
        assertThat(result.forecasts().getFirst().predictedC()).isEqualTo(70.0);
    }

    private static ThermalReading reading(Instant timestamp, Double cpuTemp) {
        return new ThermalReading(
                timestamp, true, cpuTemp, null, 30.0, 2_400.0,
                null, null, null, 40.0, "test-session");
    }
}
