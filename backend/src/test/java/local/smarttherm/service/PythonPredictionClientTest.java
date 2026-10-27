package local.smarttherm.service;

import local.smarttherm.model.ThermalReading;
import org.junit.jupiter.api.Test;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.test.web.client.MockRestServiceServer;
import org.springframework.web.client.RestClient;

import java.time.Instant;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.content;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.jsonPath;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.method;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.requestTo;
import static org.springframework.test.web.client.response.MockRestResponseCreators.withSuccess;
import static org.springframework.test.web.client.response.MockRestResponseCreators.withStatus;
import static org.springframework.http.HttpMethod.POST;

class PythonPredictionClientTest {
    @Test
    void acceptsCpuModelWhenGpuModelIsUnavailable() {
        Instant generatedAt = Instant.now();
        Instant targetTime = generatedAt.plusSeconds(300);
        RestClient.Builder builder = RestClient.builder();
        MockRestServiceServer server = MockRestServiceServer.bindTo(builder).build();
        RestClient restClient = builder.baseUrl("http://127.0.0.1:8090").build();
        server.expect(requestTo("http://127.0.0.1:8090/predict"))
                .andExpect(method(POST))
                .andExpect(content().contentTypeCompatibleWith(MediaType.APPLICATION_JSON))
                .andExpect(jsonPath("$.readings[0].cpuTemperatureC").value(61.0))
                .andRespond(withSuccess("""
                        {
                          "status":"READY",
                          "model":"Python trained models: XGBoost",
                          "trainedModel":true,
                          "generatedAt":"%s",
                          "targetTime":"%s",
                          "forecasts":[
                            {"component":"CPU","currentC":61.0,"predictedC":65.0,"generatedAt":"%s","targetTime":"%s","status":"READY","model":"XGBoost"},
                            {"component":"GPU","currentC":null,"predictedC":null,"generatedAt":"%s","targetTime":"%s","status":"MODEL_UNAVAILABLE","model":"No validated trained model is available."}
                          ]
                        }
                        """.formatted(
                        generatedAt,
                        targetTime,
                        generatedAt,
                        targetTime,
                        generatedAt,
                        targetTime), MediaType.APPLICATION_JSON));

        PythonPredictionClient client = new PythonPredictionClient(restClient, true);
        var result = client.predict(List.of(reading(generatedAt, 61.0)));

        assertThat(result.prediction()).isPresent();
        assertThat(result.prediction().orElseThrow().trainedModel()).isTrue();
        assertThat(result.prediction().orElseThrow().forecasts()).hasSize(2);
        server.verify();
    }

    @Test
    void rejectsTrainedModelResponseForADifferentCurrentReading() {
        Instant generatedAt = Instant.now();
        Instant targetTime = generatedAt.plusSeconds(300);
        RestClient.Builder builder = RestClient.builder();
        MockRestServiceServer server = MockRestServiceServer.bindTo(builder).build();
        RestClient restClient = builder.baseUrl("http://127.0.0.1:8090").build();
        server.expect(requestTo("http://127.0.0.1:8090/predict"))
                .andRespond(withSuccess("""
                        {
                          "status":"READY",
                          "model":"Python trained models: XGBoost",
                          "trainedModel":true,
                          "generatedAt":"%s",
                          "targetTime":"%s",
                          "forecasts":[
                            {"component":"CPU","currentC":70.0,"predictedC":75.0,"generatedAt":"%s","targetTime":"%s","status":"READY","model":"XGBoost"},
                            {"component":"GPU","currentC":null,"predictedC":null,"generatedAt":"%s","targetTime":"%s","status":"MODEL_UNAVAILABLE","model":"No model"}
                          ]
                        }
                        """.formatted(
                        generatedAt,
                        targetTime,
                        generatedAt,
                        targetTime,
                        generatedAt,
                        targetTime), MediaType.APPLICATION_JSON));

        var result = new PythonPredictionClient(restClient, true)
                .predict(List.of(reading(generatedAt, 61.0)));

        assertThat(result.prediction()).isEmpty();
        assertThat(result.reason()).contains("did not return a valid");
        server.verify();
    }

    @Test
    void reportsPythonRequestErrorsWithoutCallingTheServiceUnavailable() {
        RestClient.Builder builder = RestClient.builder();
        MockRestServiceServer server = MockRestServiceServer.bindTo(builder).build();
        RestClient restClient = builder.baseUrl("http://127.0.0.1:8090").build();
        server.expect(requestTo("http://127.0.0.1:8090/predict"))
                .andRespond(withStatus(HttpStatus.UNPROCESSABLE_ENTITY)
                        .contentType(MediaType.APPLICATION_JSON)
                        .body("{\"detail\":\"invalid request\"}"));

        var result = new PythonPredictionClient(restClient, true)
                .predict(List.of(reading(Instant.now(), 61.0)));

        assertThat(result.prediction()).isEmpty();
        assertThat(result.reason()).contains("HTTP 422").doesNotContain("service is unavailable");
        server.verify();
    }

    private static ThermalReading reading(Instant timestamp, double cpuTemperature) {
        return new ThermalReading(
                timestamp, true, cpuTemperature, null, 30.0, 2400.0,
                null, null, null, 45.0, "test-session");
    }
}
