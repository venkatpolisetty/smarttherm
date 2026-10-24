package local.smarttherm.api;

import local.smarttherm.config.WebConfig;
import local.smarttherm.model.AppSettings;
import local.smarttherm.model.MonitoringStatus;
import local.smarttherm.service.AlertService;
import local.smarttherm.service.CollectorProcessManager;
import local.smarttherm.service.ProcessTelemetryService;
import local.smarttherm.service.PythonPredictionClient;
import local.smarttherm.service.PredictionService;
import local.smarttherm.service.SettingsService;
import local.smarttherm.service.TelemetryService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.context.annotation.Import;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;

import static org.mockito.Mockito.when;
import static org.mockito.Mockito.verify;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.options;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.put;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.header;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@WebMvcTest(ApiController.class)
@Import(WebConfig.class)
class ApiControllerTest {
    @Autowired
    private MockMvc mockMvc;

    @MockitoBean
    private TelemetryService telemetryService;

    @MockitoBean
    private PredictionService predictionService;

    @MockitoBean
    private SettingsService settingsService;

    @MockitoBean
    private AlertService alertService;

    @MockitoBean
    private ProcessTelemetryService processTelemetryService;

    @MockitoBean
    private CollectorProcessManager collectorProcessManager;

    @MockitoBean
    private PythonPredictionClient pythonPredictionClient;

    @Test
    void returnsHealthAndDisconnectedMonitoringState() throws Exception {
        when(telemetryService.status()).thenReturn(
                new MonitoringStatus(false, false, null, "Start the collector."));

        mockMvc.perform(get("/api/health"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.status").value("UP"));
        mockMvc.perform(get("/api/monitoring/status"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.active").value(false))
                .andExpect(jsonPath("$.message").value("Start the collector."));
    }

    @Test
    void validatesSettingsRequests() throws Exception {
        when(settingsService.update(new AppSettings(90.0, 85.0, true)))
                .thenReturn(new AppSettings(90.0, 85.0, true));

        mockMvc.perform(put("/api/settings")
                        .contentType("application/json")
                        .content("""
                                {"cpuWarningC":90,"gpuWarningC":85,"notificationsEnabled":true}
                                """))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.cpuWarningC").value(90.0));

        mockMvc.perform(put("/api/settings")
                        .contentType("application/json")
                        .content("""
                                {"cpuWarningC":120,"gpuWarningC":85,"notificationsEnabled":true}
                                """))
                .andExpect(status().isBadRequest());
    }

    @Test
    void startsCollectionAndRequestsImmediateTrainingCheck() throws Exception {
        when(collectorProcessManager.startCollection()).thenReturn(true);
        when(pythonPredictionClient.requestTrainingCheck()).thenReturn(true);

        mockMvc.perform(org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post("/api/calibration/start"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.collectorStarted").value(true))
                .andExpect(jsonPath("$.trainingCheckRequested").value(true))
                .andExpect(jsonPath("$.message").value(org.hamcrest.Matchers.containsString("Calibration started")));

        verify(collectorProcessManager).startCollection();
        verify(pythonPredictionClient).requestTrainingCheck();
    }

    @Test
    void allowsBrowserPreflightForCalibrationPost() throws Exception {
        mockMvc.perform(options("/api/calibration/start")
                        .header("Origin", "http://127.0.0.1:5173")
                        .header("Access-Control-Request-Method", "POST"))
                .andExpect(status().isOk())
                .andExpect(header().string("Access-Control-Allow-Methods",
                        org.hamcrest.Matchers.containsString("POST")));
    }

    @Test
    void stopsCalibrationFromDashboard() throws Exception {
        when(collectorProcessManager.stopCollection()).thenReturn(true);

        mockMvc.perform(org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post("/api/calibration/stop"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.collectorStopped").value(true))
                .andExpect(jsonPath("$.message").value(org.hamcrest.Matchers.containsString("Calibration stopped")));

        verify(collectorProcessManager).stopCollection();
    }
}
