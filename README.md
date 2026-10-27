# SmartTherm

SmartTherm is a Windows-first, local PC thermal dashboard. It reuses the existing `ThermalCollector` .NET application and its LibreHardwareMonitor/HWiNFO integrations for real hardware readings, exposes them through a loopback-only Java 21 / Spring Boot API, and presents a React dashboard. It does not make fan, CPU, or other hardware changes.

The current workspace has no validated trained model. Historical CSVs are not presented as live readings, and SmartTherm does not show forecast temperatures until local trained-model inference is ready.

## Requirements

- Windows 10/11
- Java 21
- Maven 3.9+
- .NET 10 SDK (the existing collector targets `net10.0-windows`)
- Node.js 20+ and npm
- 64-bit Python 3.12+ for the ML assessment and local inference service
- LibreHardwareMonitor-supported sensor hardware; HWiNFO is optional and must be running for its shared-memory readings

The collector may need to be run elevated for some sensors. Sensor exposure depends on the device, drivers, firmware, and permissions.

## Run locally

Open PowerShell in the existing BTP folder (the folder containing this README):

1. Build and start the API (which also starts sensor collection). Run both commands from the BTP folder:

   ```powershell
   mvn -f .\backend\pom.xml verify
   java -jar .\backend\target\smarttherm-api-1.0.0.jar
   ```

   Run the JAR from the BTP folder so its working directory resolves the existing `ThermalCollector` project and local data paths correctly. Spring starts and supervises the .NET sensor collector automatically. The collector samples approximately every five seconds, writes its current reading, bounded JSONL history, and versioned/session-aware training CSVs under `ThermalCollector\data\`, and stops when the API shuts down.

2. In another terminal, start the local Python inference service:

   ```powershell
   powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\start-ml-service.ps1
   ```

   It creates `ml-integration\.venv` and installs the declared requirements on first run. Without this service, the API reports predictions unavailable.

3. In another terminal, install frontend dependencies once, then start Vite:

   ```powershell
   npm.cmd --prefix .\frontend install
   npm.cmd --prefix .\frontend run dev -- --host 127.0.0.1
   ```

4. Open <http://127.0.0.1:5173>.

Alternatively, run this from PowerShell in the project folder to start the backend first, then the frontend and local Python inference service in separate PowerShell windows:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\start-smarttherm.ps1
```

The process-scoped execution-policy flag lets the script run without changing your machine's persistent policy. Before opening service windows, the launcher checks Java 21, .NET 10, Node.js 20+, Python 3.12+, and Maven 3.9+; it checks that application ports are free and runs `mvn -f .\backend\pom.xml verify`. It stops and prints Maven's build output if verification fails, so no service windows are started after a failed backend build. It waits for the backend API and then all three local HTTP endpoints before reporting that the dashboard is ready. The backend owns collector startup and shutdown, avoiding a second collection loop. The first Python service start creates `ml-integration\.venv` and installs its requirements. Close the backend window to stop the collector; close the other service windows to stop Python inference and Vite.

The launcher uses Maven 3.9+ from `PATH`, or an already cached Maven distribution under `%USERPROFILE%\.m2\wrapper\dists`. If neither is available, install Maven and add its `bin` folder to `PATH`, then open a new PowerShell window. For a direct build diagnostic, run `mvn -f .\backend\pom.xml verify` from this folder and follow the first reported error. The backend targets Java 21. Starting the packaged JAR from the BTP folder also ensures that relative collector and storage paths point to the project root.

The Java API binds only to `127.0.0.1:8080`, Python inference to `127.0.0.1:8090`, and Vite to `127.0.0.1:5173`. These local services do not expose a public network listener.

## Architecture and data

- **Sensor bridge:** `ThermalCollector\Program.cs` currently uses the `LibreHardwareMonitorLib` API directly; it does not connect to a `localhost:8085` HTTP JSON endpoint. It retains the existing HWiNFO shared-memory temperature supplement because CPU/iGPU readings may not be exposed by LibreHardwareMonitor on this hardware. Available CPU power and CPU fan readings are also exposed. Unavailable metrics are written as JSON `null`; they are never substituted with sample readings. The collector prevents a second instance, retries hardware initialization, and recovers from per-sample sensor errors. The current snapshot is atomically replaced at `ThermalCollector\data\current-reading.json`. JSONL sensor history is bounded. System and process CSVs, as well as versioned training CSVs, rotate at 10 MiB and retain up to approximately 512 MiB per file group.
- **CPU throttle calibration:** Use **Start CPU throttle calibration** in the dashboard. It runs a CPU load for up to 300 seconds while sampling CPU temperature, load, clock, package power, and fresh HWiNFO shared-memory indicators when available. It stops if CPU temperature reaches the 95°C safety ceiling or both clock and hardware throttle indicators are unavailable. A hardware-reported throttle flag can confirm an event under high load without requiring a particular temperature or clock sensor. Without a flag, suspected events require a sustained clock drop to 85% or less of the high-load baseline alongside a relative temperature rise. Per-event temperatures, their median, method, event count, baseline, and timestamp are saved to `smarttherm_profile.json`; no event produces a null temperature and `not_determined`. Progress and result are available from `/api/calibration/status`. HWiNFO shared-memory support should be enabled for hardware confirmation.
- **API:** Java 21 and Spring Boot start and supervise the collector, poll the current snapshot, validate metrics, retain recent chart history, and maintain a separate timestamp-based 300-second active inference window. Settings and up to 100 alert events are persisted as local JSON files in `backend\data\`.
- **Prediction:** Offline Python tooling assesses genuine five-minute temperature targets and compares direct XGBoost, residual XGBoost, and Gradient Boosting against persistence using chronological time-block validation/test splits. The residual model predicts temperature change, with correction strength selected on validation data. Labels use actual later readings about 300 seconds after feature time; models are saved only if they beat the baseline on both MAE and RMSE. The loopback Python service loads installed local artifacts; Java validates them and calls the service with the rolling window and bounded timeouts. If no validated model or fresh, sufficient input is available, prediction status is reported as not ready and predicted temperatures remain unavailable. No model is retrained in the live monitoring loop, and no accuracy/confidence is claimed without evidence.
- **Alerts:** User-configurable CPU and GPU warning thresholds default to 90°C and 85°C; these are warning settings, not claimed hardware throttle limits. A warning is created only when a trained five-minute prediction crosses its configured threshold; critical is 5°C above that threshold. An alert remains active until its prediction falls 2°C below the warning threshold. Repeated events are suppressed while risk remains active and throttled for ten minutes after resolution. Browser notifications are best effort and require user permission; in-app alerts remain available if notifications are unavailable or denied.
- **Frontend:** React, Vite, and Recharts. It uses API data only, shows disconnected/empty/error states, and does not include demo sensor readings.
- **Privacy:** Thermal data and settings remain local. Process names/paths in the existing collector CSVs are not consumed by the dashboard or API.

For the current CSV assessment, real-data collection guidance, model-training commands, evaluation gates, and local inference behavior, see `ml-integration\README.md`. Assessment output includes the per-component examples that survive chronological splitting and the five-minute embargo; sufficient split counts do not guarantee promotion because a candidate must also beat the persistence baseline.

## Assess and train models

Start the API (the collector starts automatically) and collect continuous real sensor data first. The local Python inference service checks historical training data once at startup and periodically thereafter (every six hours by default). It trains in a background worker, separately from five-second collection and prediction, and promotes a model only when it passes the existing chronological validation/test and persistence-baseline gates. New or changed CSV data is checked at the next scheduled run; insufficient or unchanged data is not retrained repeatedly.

Initialize the Python environment and inspect usable labels:

```powershell
python -m venv .\ml-integration\.venv
& .\ml-integration\.venv\Scripts\python.exe -m pip install -r .\ml-integration\requirements.txt
& .\ml-integration\.venv\Scripts\python.exe .\ml-integration\assess_data.py
```

Automatic training can be disabled with `SMARTTHERM_AUTO_TRAINING=false` or its interval changed with `SMARTTHERM_TRAINING_CHECK_INTERVAL_SECONDS` (minimum 60 seconds). To run the same gated training pipeline manually after assessment:

```powershell
& .\ml-integration\.venv\Scripts\python.exe .\ml-integration\train_models.py
```

Current workspace data is insufficient for validated models: the assessor finds zero CPU targets, and the GPU examples fail the chronological split requirements. Automatic training reports this state without inventing predictions. See the ML integration guide for data requirements and model-retention policy. The backend build passed `mvn verify` with Java 21 and Maven 3.10.0 from the local Maven cache; the earlier launch environment did not have `mvn` available on `PATH`.

## Planned design (not implemented)

The target design is per-device calibration, an RC thermal physics model, and a gradient-boosting residual correction on top of it. None of this is implemented yet. The current predictor is the XGBoost / Gradient Boosting approach described above, and a model is only used if it beats the persistence baseline. "CPU throttle calibration" in this README refers to throttle-temperature detection, not per-device model calibration.

## API

All endpoints are local at `http://127.0.0.1:8080`.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | API health |
| GET | `/api/monitoring/status` | Collector connectivity and last accepted sample |
| GET | `/api/temperature/current` | Most recent valid sensor reading |
| GET | `/api/temperature/history?limit=600` | Recent readings; limit is bounded to 2,000 |
| GET | `/api/predictions/latest` | Latest five-minute predictions or explicit model/readiness status |
| GET | `/api/alerts` | Recent persisted thermal alert events |
| POST | `/api/calibration/start` | Start the bounded CPU throttle calibration |
| GET | `/api/calibration/status` | Calibration state and latest detected result |
| POST | `/api/calibration/stop` | Stop calibration and collector |
| GET | `/api/settings` | Threshold and notification settings |
| PUT | `/api/settings` | Validate and save settings |

Example settings request:

```json
{
  "cpuWarningC": 90.0,
  "gpuWarningC": 85.0,
  "notificationsEnabled": true
}
```

Thresholds must be between 50°C and 110°C. `backend\data\settings.example.json` is an example; the application creates its own `settings.json` when the user saves settings. The local data files are ignored by Git.

## Limitations

- Depends on third-party sensor tooling (LibreHardwareMonitorLib, optionally HWiNFO). Some CPU/iGPU temperatures are only available through HWiNFO shared memory.
- No validated trained model exists yet, so no real predictions or alerts are produced.
- Sensor access may require running elevated; exposure varies by device, drivers, and firmware.

## Troubleshooting

- **Collector disconnected:** Confirm the backend can find `dotnet` and `ThermalCollector\ThermalCollector.csproj`, then check its collector startup/error logs and `ThermalCollector\data\current-reading.json`. Set `SMARTTHERM_COLLECTOR_ENABLED=false` only when intentionally disabling automatic collection.
- **No temperature shown:** The hardware or driver may not expose a temperature sensor through LibreHardwareMonitor. Check the collector's sensor diagnostic output. HWiNFO can provide some additional CPU/iGPU readings when its shared-memory support is enabled.
- **No prediction:** A forecast is withheld until a validated local model and sufficient fresh input history are available. Ten minutes of collection by itself does not mean a model has been trained; run the documented assessment and training steps.
- **Port in use:** Stop the existing local process using port 8080 (API) or 5173 (Vite) before starting SmartTherm.
- **`mvn`, `dotnet`, or `npm` not found:** Install the prerequisites and reopen PowerShell so PATH changes take effect.
- **Python reports insufficient training data:** Keep collecting continuous samples with the actual CPU/GPU temperature sensor available. Training refuses missing features/labels and gapped sessions; do not treat that refusal as a successful model build.
- **No trained model is installed:** Python reports each component model as unavailable. The dashboard explicitly shows **Model not ready** for that component and leaves its prediction blank; SmartTherm does not use an untrained linear estimate.
- **Launcher stops before startup:** Follow the missing or invalid prerequisite reported in the launcher window. If Maven verification fails, fix the first Maven error and rerun the launcher; it deliberately starts no services after a failed backend build.
- **Backend is healthy but collection is disconnected:** Check the backend window for collector startup errors, verify .NET 10 and the project path, then restart the backend to let Spring start the single managed collector again. The collector's mutex prevents a second collection loop.
- **API is healthy but Python inference is unavailable:** Check `http://127.0.0.1:8090/health` and the Python service window. Restart the Python service with `.\scripts\start-ml-service.ps1`; a `MODEL_UNAVAILABLE` result is expected until a validated model is promoted.
- **Stopping SmartTherm:** Close the backend window first so Spring can send the collector its clean stop request. Then close the frontend and Python service windows. If the collector does not exit cleanly, Spring terminates its process tree and reports the shutdown issue.
- **Browser notifications:** Allow permission through the dashboard's notification control. Browser permission denial does not disable in-app alerts.

## Tests and builds

```powershell
mvn -f .\backend\pom.xml test
npm.cmd --prefix .\frontend install
npm.cmd --prefix .\frontend run build
dotnet build .\ThermalCollector\ThermalCollector.csproj
```

Sensor integration needs Windows hardware and is not fully exercised by unit tests. The prediction and alert tests use isolated test inputs; these values are not part of the production dashboard.
