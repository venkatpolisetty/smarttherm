# SmartTherm ML training and inference

## Current model status

Previously promoted ten-minute artifacts are incompatible with the five-minute model contract and are ignored. `ThermalCollector\ml_test.py` only prints CSV shape and missing-value counts; it does not train or predict.

Re-run the assessment command below for five-minute label counts. Earlier row and split counts were computed for ten-minute targets and do not describe this model contract. Do not interpret old recorded CSV readings as live sensor data.

Until new data passes the checks below and a validated artifact is trained, SmartTherm reports that the model is not ready and leaves predicted temperatures unavailable. It does not substitute a linear trend or another untrained estimate.

## Collect real training data

Prerequisites: 64-bit Python 3.12+ and the .NET 10 SDK. Starting the Spring backend automatically starts the collector. For collection without the dashboard/API, it can also be run directly:

```powershell
dotnet run --project .\ThermalCollector\ThermalCollector.csproj
```

Each collector run now appends a versioned CSV such as:

```text
ThermalCollector\data\training_telemetry_<session-id>_part000.csv
```

The file has a UTC timestamp and stable columns for session ID, CPU/GPU temperature, utilization, CPU clock, CPU power, CPU fan speed, memory load, and GPU sensor identity. Sensor fields unavailable on your hardware are blank, never filled with fabricated values. A collector restart begins a new session. Files roll at about 10 MiB; old training files are pruned toward a 512 MiB total retention budget.

CPU or GPU temperature data is hardware- and driver-dependent. Check the collector's sensor diagnostics and leave HWiNFO running if its local shared-memory readings are required. Sampling is about every five seconds. Collect across real workloads and idle periods, and avoid stopping the collector during useful sessions. A model component needs continuous samples around ten minutes apart; collection time alone is not sufficient when sensor metrics are missing or large gaps split a session.

## Set up Python and assess data

In PowerShell from the BTP folder:

```powershell
python -m venv .\ml-integration\.venv
& .\ml-integration\.venv\Scripts\python.exe -m pip install -r .\ml-integration\requirements.txt
& .\ml-integration\.venv\Scripts\python.exe .\ml-integration\assess_data.py
```

For a machine-readable report:

```powershell
& .\ml-integration\.venv\Scripts\python.exe .\ml-integration\assess_data.py --json
```

The assessor reports usable five-minute pairs and the eligible chronological train, validation, and test counts per component using the same 300/50/50 minimums and embargo as training. It normalizes collector CSV column names but does not join separate CSV sessions, invent missing temperatures, or interpolate across gaps. It requires a real later sensor observation approximately 300 seconds after each feature timestamp (within 15 seconds), continuous same-session samples with at most 20 seconds between readings, valid temperature bounds, and enough history for features. CPU and GPU labels are assessed independently.

## Train and evaluate

### Synthetic development data (not live model data)

The supplied CSVs do not have enough real CPU temperature time series to train and validate a five-minute CPU model. To exercise the feature contract and training code, run:

```powershell
& .\ml-integration\.venv\Scripts\python.exe .\ml-integration\generate_synthetic.py
```

This writes `ml-integration\synthetic-lab\synthetic_thermal_data.csv`, a separate 5-second, UTC CSV with the exact versioned ThermalCollector training schema. It contains 12 one-hour synthetic sessions with mostly idle/normal readings plus ramp, sustained load, mixed load, and cooling scenarios. Temperatures follow a lagged load-and-cooling simulation. Observed real data anchors available ranges, but CPU temperature is present in only six parsed real rows and CPU clock is absent; therefore hot-load behavior, CPU clock, power, and fan relationships are assumptions, not measured calibration. Synthetic median values can differ from the mostly idle real data and must not be read as a measured hardware profile.

The command checks schema, duplicates, missing/invalid parsed rows, five-second intervals, temperature step sizes, feature ranges, and selected load/temperature correlations. It evaluates Gradient Boosting and XGBoost on chronological synthetic splits against persistence, selects by validation RMSE, and reports the untouched test split separately. It saves that research candidate under `synthetic-lab\artifacts\`, with `evaluation.json` beside the CSV. These artifacts are intentionally outside `models\`; they are never loaded by the application or used for live predictions. Synthetic metrics test software behavior only. A real CPU model remains unavailable until sufficient real, continuous CPU-temperature sessions pass the normal assessment and real holdout gates.

The local inference service automatically checks for changed training data at startup and every 15 minutes by default. It runs training in a background worker, independent of five-second sensor collection and prediction. A persisted fingerprint prevents repeatedly training on unchanged CSV files; when collection adds or changes data, the next scheduled check reassesses it. Once the local collector has gathered enough continuous, real CPU or GPU temperature history to form five-minute targets and chronological train/validation/test partitions, the trainer fits candidates and evaluates them. A candidate replaces that component's model only when it beats persistence on validation and held-out real data and does not regress the active model. The model registry notices a promoted manifest on its next health or prediction request and reloads the artifact. Insufficient data, rejected candidates, and training errors are reported through `GET /health` under `training`; the service continues running and does not fabricate a model or prediction.

Automatic training can be disabled with `SMARTTHERM_AUTO_TRAINING=false`. Set `SMARTTHERM_TRAINING_CHECK_INTERVAL_SECONDS` to change the interval (minimum 60 seconds). Only one training job may use a model directory at a time, including a manual CLI run. The shared training pipeline preserves the existing data requirements and promotion gates.

To assess labels manually, run:

```powershell
& .\ml-integration\.venv\Scripts\python.exe .\ml-integration\assess_data.py
```

The shared pipeline compares direct XGBoost, residual XGBoost, and scikit-learn Gradient Boosting. The residual XGBoost predicts temperature change from the current reading; its correction strength is selected on validation data and applied to persistence. All candidates use lag/window/trend features available at prediction time, median imputers fitted on the training partition only, chronological one-hour session blocks, and a five-minute embargo between train/validation/test boundaries. The trainer requires at least 300 training examples and 50 validation plus 50 held-out test examples after splitting. A candidate is saved only if it improves on the persistence baseline on both MAE and RMSE on validation and held-out test data, and does not regress the active model on that held-out split. After passing those checks, the deployed pipeline is refit on every eligible example collected so far; its recorded metrics remain from the untouched chronological holdouts. Otherwise the report says `NOT_PROMOTED` or `INSUFFICIENT_DATA`; SmartTherm continues to report no trained prediction rather than substituting an untrained estimate. Do not lower the evidence thresholds to make an underpowered dataset appear trainable.

An optional manual run uses the same safeguards:

```powershell
& .\ml-integration\.venv\Scripts\python.exe .\ml-integration\train_models.py
```

Accepted models and a `manifest.json` are stored locally in `ml-integration\models\`, ignored by Git. Training is not run per reading and does not block live collection or inference. Models are loaded with `joblib`, so only load locally generated/trusted artifacts. The inference service checks for a newly promoted manifest during health and prediction requests and reloads it while keeping the working model if a reload fails.

## Start the local inference service

The all-services launcher starts the Python service in a separate PowerShell window. To run it on its own:

```powershell
.\scripts\start-ml-service.ps1
```

The script creates `ml-integration\.venv` if needed, installs the declared dependencies, then starts Uvicorn on **127.0.0.1:8090**. Model artifacts are read once from the local `ml-integration\models\` directory. No cloud, public-network listener, database, or Docker is used.

Endpoints:

- `GET /health`: reports API health, per-component artifact availability, and automatic-training status.
- `POST /training/check`: queues an immediate background check after the dashboard's **Start calibration** action. It does not bypass the real-data or model-promotion gates.
- `POST /predict`: accepts up to 150 chronologically ordered, timezone-stamped real readings from the latest 300 seconds. Out-of-range, stale, missing, or gapped inputs are rejected or returned as an explicit unavailable status.

The Java API maintains this active 300-second window separately from historical training CSVs, calls `/predict` with bounded timeouts, and validates response freshness, component readings, model metadata, and temperature range. A component forecast is used only for a valid `READY` trained-model result. If Python or a model is unavailable, predicted temperatures remain unavailable; this does not prevent independent sensor collection. Existing configured thresholds, risk state, and alert de-duplication are applied only to available trained predictions.

## Data contract and evaluation

- Telemetry schema version: `1`; model artifact schema version: `2`; timestamps are UTC; temperatures are Celsius; clock speed is MHz; load is percent.
- Five-minute target: actual same-component temperature sampled 300 seconds later ±15 seconds in the same continuous session. Gaps over 20 seconds break the segment.
- Component models train independently. Missing CPU data does not block a GPU assessment; a GPU model is omitted if its sensor/target data is inadequate.
- Features include current/mean/min/max temperatures, 30/60/300-second trends, CPU/GPU load, CPU clock, and memory utilization when available. Missing feature values are handled by the saved training pipeline.
- Reports include sample counts and per-component MAE/RMSE; these are not a guarantee of future accuracy.

The Python tests are synthetic unit fixtures for schema, target, time-split, and API behavior; they are not training data or product-performance evidence. Actual hardware/model efficacy remains unverified until the collector records sufficient real telemetry and a candidate passes the held-out test.
