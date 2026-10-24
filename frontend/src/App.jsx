import { useCallback, useEffect, useRef, useState } from "react";
import {
  Activity,
  AlertTriangle,
  Bell,
  BellOff,
  Check,
  ChevronDown,
  Cpu,
  Fan,
  Gauge,
  HardDrive,
  Play,
  Radio,
  RefreshCw,
  Settings2,
  Square,
  ShieldCheck,
  Thermometer,
  TriangleAlert,
  Zap,
} from "lucide-react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const REFRESH_MS = 5000;
const DEMO_SCENARIOS = [
  { id: "normal", label: "Normal", cpuOffset: -26, predictedOffset: -24, utilization: 23, trend: "Stable", risk: "normal", severity: "NORMAL", action: "No action needed. Continue normal monitoring." },
  { id: "elevated", label: "Elevated temperature", cpuOffset: -12, predictedOffset: -6, utilization: 74, trend: "Rising steadily", risk: "elevated", severity: "ELEVATED", action: "Check that airflow is clear and reduce sustained workload if the temperature keeps rising." },
  { id: "high-risk", label: "High risk", cpuOffset: -2, predictedOffset: 2, utilization: 91, trend: "Rising quickly", risk: "warning", severity: "WARNING", action: "Consider reducing CPU-intensive work and check that cooling is operating normally." },
  { id: "critical", label: "Critical risk", cpuOffset: 4, predictedOffset: 8, utilization: 96, trend: "Rising quickly", risk: "critical", severity: "CRITICAL", action: "Stop intensive work and allow the system to cool. Check cooling and airflow." },
  { id: "sensor-unavailable", label: "Sensor unavailable", cpu: null, predicted: null, utilization: null, trend: "Unavailable", risk: "unavailable", severity: "UNAVAILABLE", action: "Check sensor access and collector status. No temperature is being assumed." },
  { id: "prediction-unavailable", label: "Prediction unavailable", cpuOffset: -14, predicted: null, utilization: 48, trend: "Not available", risk: "unavailable", severity: "UNAVAILABLE", action: "Current readings remain visible, but no forecast is shown until model inference is available." },
];

function initialDemoScenario() {
  const requested = new URLSearchParams(window.location.search).get("demo");
  if (requested === "warning") return "high-risk";
  return DEMO_SCENARIOS.some((scenario) => scenario.id === requested) ? requested : null;
}

async function getJson(path) {
  const response = await fetch(path);
  if (response.status === 204) return null;
  if (!response.ok) throw new Error(`API request failed (${response.status})`);
  return response.json();
}

function formatTemp(value) {
  return value == null ? "--" : `${Number(value).toFixed(1)}°C`;
}

function formatTime(value) {
  if (!value) return "Not available";
  return new Date(value).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function statusFor(reading, threshold) {
  if (reading == null) return "unavailable";
  if (reading >= threshold + 5) return "critical";
  if (reading >= threshold) return "warning";
  return "normal";
}

function forecastReadiness(forecast) {
  if (forecast?.status === "READY") return "Model ready";
  if (forecast?.status === "MODEL_UNAVAILABLE") return "Model not ready";
  if (forecast?.status === "INSUFFICIENT_HISTORY") return "Insufficient sensor history";
  if (forecast?.status === "MODEL_ERROR") return "Model error";
  if (forecast?.status === "STALE_DATA") return "Waiting for fresh sensor data";
  return "Waiting for model status";
}

function StatCard({ title, value, subtitle, icon: Icon, status = "normal", accent }) {
  return (
    <article className={`stat-card ${status}`} style={{ "--accent": accent }}>
      <div className="stat-card-head">
        <span className="stat-icon"><Icon size={17} strokeWidth={1.8} /></span>
        <span className={`status-pill ${status}`}><i />{status}</span>
      </div>
      <div className="stat-title">{title}</div>
      <div className="stat-value">{value}</div>
      <div className="stat-subtitle">{subtitle}</div>
    </article>
  );
}

function App() {
  const [demoScenario, setDemoScenario] = useState(initialDemoScenario);
  const warningDemo = demoScenario !== null;
  const demoModeRef = useRef(warningDemo);
  demoModeRef.current = warningDemo;
  const [current, setCurrent] = useState(null);
  const [history, setHistory] = useState([]);
  const [status, setStatus] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [settings, setSettings] = useState(null);
  const [processUsage, setProcessUsage] = useState(null);
  const [endingPid, setEndingPid] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [calibrationStarting, setCalibrationStarting] = useState(false);
  const [calibrationStatus, setCalibrationStatus] = useState(null);
  const [calibrationMessage, setCalibrationMessage] = useState("");
  const [permission, setPermission] = useState(
    "Notification" in window ? Notification.permission : "unsupported",
  );
  const notifiedIds = useRef(new Set());

  const refresh = useCallback(async () => {
    try {
      const [nextStatus, nextCurrent, nextHistory, nextPrediction, nextAlerts, nextSettings, nextProcesses, nextCalibration] =
        await Promise.all([
          getJson("/api/monitoring/status"),
          getJson("/api/temperature/current"),
          getJson("/api/temperature/history?limit=240"),
          getJson("/api/predictions/latest"),
          getJson("/api/alerts"),
          getJson("/api/settings"),
          getJson("/api/processes"),
          getJson("/api/calibration/status"),
        ]);
      if (demoModeRef.current) return;
      setStatus(nextStatus);
      setCurrent(nextCurrent);
      setHistory(nextHistory);
      setPrediction(nextPrediction);
      setAlerts(nextAlerts);
      setSettings(nextSettings);
      setProcessUsage(nextProcesses);
      setCalibrationStatus(nextCalibration);
      setError("");
      if (nextSettings.notificationsEnabled && "Notification" in window &&
          Notification.permission === "granted") {
        nextAlerts.filter((alert) => alert.active).forEach((alert) => {
          if (notifiedIds.current.has(alert.id)) return;
          notifiedIds.current.add(alert.id);
          new Notification(`SmartTherm ${alert.severity.toLowerCase()}`, {
            body: alert.message,
            tag: alert.component,
          });
        });
      }
    } catch (requestError) {
      if (!demoModeRef.current) setError(requestError.message || "Could not connect to the SmartTherm API.");
    } finally {
      setLoading(false);
    }
  }, []);

  async function startCalibration() {
    setCalibrationStarting(true);
    setCalibrationMessage("");
    try {
      const response = await fetch("/api/calibration/start", { method: "POST" });
      if (!response.ok) throw new Error(`Could not start calibration (${response.status}).`);
      const result = await response.json();
      setCalibrationMessage(
        `${result.message}${result.trainingCheckRequested ? " An immediate training-data check was queued." : " The ML service is unavailable; automatic checks will resume when it is running."}`,
      );
      await refresh();
    } catch (requestError) {
      setCalibrationMessage(requestError.message || "Could not contact SmartTherm. Start the local application first.");
    } finally {
      setCalibrationStarting(false);
    }
  }

  async function stopCalibration() {
    setCalibrationStarting(true);
    setCalibrationMessage("");
    try {
      const response = await fetch("/api/calibration/stop", { method: "POST" });
      if (!response.ok) throw new Error(`Could not stop calibration (${response.status}).`);
      const result = await response.json();
      setCalibrationStatus({ collecting: !result.collectorStopped });
      setCalibrationMessage(result.message);
      await refresh();
    } catch (requestError) {
      setCalibrationMessage(requestError.message || "Could not contact SmartTherm.");
    } finally {
      setCalibrationStarting(false);
    }
  }

  useEffect(() => {
    if (warningDemo) {
      const now = new Date();
      const inTenMinutes = new Date(now.getTime() + 10 * 60 * 1000);
      const scenario = DEMO_SCENARIOS.find((item) => item.id === demoScenario) || DEMO_SCENARIOS[0];
      const threshold = settings?.cpuWarningC ?? 90;
      const cpu = scenario.cpuOffset == null ? scenario.cpu : threshold + scenario.cpuOffset;
      const predicted = scenario.predictedOffset == null ? scenario.predicted : threshold + scenario.predictedOffset;
      const startTemp = cpu == null ? null : Math.max(30, cpu - (scenario.trend.includes("Rising") ? 4.5 : 0.4));
      const readings = Array.from({ length: 24 }, (_, index) => {
        const timestamp = new Date(now.getTime() - (23 - index) * 5_000);
        return {
          timestamp: timestamp.toISOString(),
          cpuTemperatureC: startTemp == null ? null : Number((startTemp + (cpu - startTemp) * index / 23 + Math.sin(index / 2) * 0.15).toFixed(1)),
          gpuTemperatureC: scenario.id === "sensor-unavailable" ? null : Number((43 + Math.sin(index / 3) * 1.2).toFixed(1)),
        };
      });
      const generatedAt = now.toISOString();
      const sensorAvailable = scenario.id !== "sensor-unavailable";
      const predictionAvailable = predicted != null;
      const forecastStatus = predictionAvailable ? "READY" : scenario.id === "prediction-unavailable" ? "MODEL_UNAVAILABLE" : "INSUFFICIENT_HISTORY";
      const currentCpu = sensorAvailable ? cpu : null;
      const alert = scenario.risk === "normal" ? [] : [{
        id: `demo-${scenario.id}`,
        component: "CPU",
        severity: scenario.severity,
        active: true,
        createdAt: generatedAt,
        message: scenario.id === "sensor-unavailable"
          ? "DEMO — CPU temperature sensor is unavailable; no temperature or forecast is being shown."
          : scenario.id === "prediction-unavailable"
            ? "DEMO — CPU prediction is unavailable; current sensor readings are still available."
            : `DEMO — ${scenario.label}: CPU is ${scenario.trend.toLowerCase()} at ${cpu}°C${predicted == null ? "" : `, with a ${predicted}°C forecast in 10 minutes`}.`,
        suggestedAction: `Demo recommendation: ${scenario.action}`,
      }];
      setStatus({ active: sensorAvailable, message: sensorAvailable ? "Simulated sensor feed" : "Demo sensor unavailable" });
      setCurrent({
        timestamp: generatedAt,
        cpuTemperatureC: currentCpu,
        gpuTemperatureC: sensorAvailable ? 44.2 : null,
        cpuUtilizationPercent: sensorAvailable ? scenario.utilization : null,
        cpuClockMhz: sensorAvailable ? 3200 + scenario.utilization * 11 : null,
        cpuPowerW: sensorAvailable ? 24 + scenario.utilization * .64 : null,
        cpuFanSpeedRpm: sensorAvailable ? 1200 + Math.max(0, cpu - 45) * 45 : null,
        memoryUtilizationPercent: sensorAvailable ? 47 : null,
      });
      setHistory(readings);
      setPrediction({
        status: predictionAvailable ? "READY" : "MODEL_UNAVAILABLE",
        model: "Simulated warning demo",
        generatedAt,
        targetTime: inTenMinutes.toISOString(),
        forecasts: [
          { component: "CPU", currentC: currentCpu, predictedC: predicted, status: forecastStatus, model: "Simulated demo value" },
          { component: "GPU", currentC: sensorAvailable ? 44.2 : null, predictedC: sensorAvailable ? 45.1 : null, status: sensorAvailable ? "READY" : "INSUFFICIENT_HISTORY", model: "Simulated demo value" },
        ],
      });
      setAlerts(alert);
      setSettings((existing) => existing || { cpuWarningC: 90, gpuWarningC: 85, notificationsEnabled: false });
      setProcessUsage({
        timestamp: generatedAt, available: scenario.id !== "sensor-unavailable",
        message: "Simulated demo processes; no live apps are read or controlled.",
        processes: scenario.id === "sensor-unavailable" ? [] : [
          { processId: 1204, processName: "Browser (demo)", cpuUsagePercent: scenario.utilization ?? 0, memoryMb: 812, foreground: true },
          { processId: 2316, processName: "Editor (demo)", cpuUsagePercent: 7.4, memoryMb: 420, foreground: false },
        ],
      });
      setLoading(false);
      return undefined;
    }
    refresh();
    const timer = window.setInterval(refresh, REFRESH_MS);
    return () => window.clearInterval(timer);
  }, [refresh, demoScenario]);

  const updateSettings = async (changes) => {
    if (!settings) return;
    if (warningDemo) {
      setSettings({ ...settings, ...changes });
      return;
    }
    setSaving(true);
    try {
      const response = await fetch("/api/settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...settings, ...changes }),
      });
      if (!response.ok) throw new Error(`Could not save settings (${response.status}).`);
      setSettings(await response.json());
      setError("");
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setSaving(false);
    }
  };

  const requestNotificationPermission = async () => {
    if (warningDemo) return;
    if (!("Notification" in window)) {
      setPermission("unsupported");
      return;
    }
    setPermission(await Notification.requestPermission());
  };

  const endApp = async (process) => {
    if (warningDemo) return;
    const accepted = window.confirm(
      `End ${process.processName} (PID ${process.processId})? It may lose unsaved work. SmartTherm will ask Windows to close it.`
    );
    if (!accepted) return;
    setEndingPid(process.processId);
    try {
      const response = await fetch(`/api/processes/${process.processId}/terminate`, { method: "POST" });
      if (!response.ok) {
        const detail = await response.json().catch(() => ({}));
        throw new Error(detail.message || `Could not end app (${response.status}).`);
      }
      setError("");
      await refresh();
    } catch (requestError) {
      setError(requestError.message || "Could not end the selected app.");
    } finally {
      setEndingPid(null);
    }
  };

  const cpuForecast = prediction?.forecasts?.find((item) => item.component === "CPU");
  const gpuForecast = prediction?.forecasts?.find((item) => item.component === "GPU");
  const cpuModelReady = cpuForecast?.status === "READY";
  const gpuModelReady = gpuForecast?.status === "READY";
  const cpuThreshold = settings?.cpuWarningC ?? 90;
  const gpuThreshold = settings?.gpuWarningC ?? 85;
  const cpuState = statusFor(current?.cpuTemperatureC, cpuThreshold);
  const gpuState = statusFor(current?.gpuTemperatureC, gpuThreshold);
  const activeAlerts = alerts.filter((alert) => alert.active);
  const chartData = history.map((reading) => ({
    time: new Date(reading.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    cpu: reading.cpuTemperatureC ?? undefined,
    gpu: reading.gpuTemperatureC ?? undefined,
  }));
  const hasTemperatureHistory = chartData.filter((reading) => reading.cpu != null || reading.gpu != null).length > 1;
  const predictionLabel = warningDemo
    ? demoScenario === "prediction-unavailable"
      ? "Simulated demo · prediction unavailable (no forecast is fabricated)"
      : demoScenario === "sensor-unavailable"
        ? "Simulated demo · sensor unavailable"
        : "Simulated demo forecast · 10-minute horizon (not a live model prediction)"
    : cpuModelReady
      ? "Validated trained model · 10-minute forecast"
    : prediction?.status === "STALE_DATA"
      ? "Waiting for fresh sensor data"
      : cpuForecast?.status === "INSUFFICIENT_HISTORY" || prediction?.status === "INSUFFICIENT_HISTORY"
        ? "Collecting enough valid sensor history"
        : cpuForecast?.status === "MODEL_UNAVAILABLE" || prediction?.status === "MODEL_NOT_READY"
          ? "No validated CPU model is available · CPU predictions are not shown"
          : "Waiting for model readiness";

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand"><span className="brand-mark"><Thermometer size={19} /></span><span>smart<span>therm</span></span></div>
        <div className="nav-label">WORKSPACE</div>
        <a className="nav-item active" href="#overview"><Gauge size={17} /> Overview</a>
        <a className="nav-item" href="#history"><Activity size={17} /> Temperature history</a>
        <a className="nav-item" href="#alerts"><Bell size={17} /> Alert center
          {activeAlerts.length > 0 && <span className="nav-count">{activeAlerts.length}</span>}
        </a>
        <div className="sidebar-bottom">
          <div className="local-card">
            <span className="local-icon"><ShieldCheck size={16} /></span>
            <div><b>Private by design</b><span>Data stays on this PC</span></div>
          </div>
          <div className="sidebar-version">SMARTTHERM <span>·</span> LOCAL MONITOR</div>
        </div>
      </aside>

      <section className="main-content" id="overview">
        <header className="topbar">
          <div className="breadcrumb">Monitoring <span>/</span> <b>Overview</b></div>
          <div className="topbar-right">
            <span className={`connection ${status?.active ? "connected" : "disconnected"}`}>
              <i />{warningDemo ? "Demo data" : status?.active ? "Collector connected" : "Collector disconnected"}
            </span>
            <button className="icon-button" onClick={refresh} aria-label="Refresh data" disabled={warningDemo}><RefreshCw size={16} /></button>
            <div className="avatar">ST</div>
          </div>
        </header>

        <div className="dashboard">
          {warningDemo && <div className="demo-banner" role="note">
            <TriangleAlert size={16} />
            <span><b>DEMO MODE</b> — simulated values only; no live readings, model changes, or OS notifications.</span>
            <button className="demo-exit" onClick={() => {
              const url = new URL(window.location.href);
              url.searchParams.delete("demo");
              window.history.replaceState({}, "", `${url.pathname}${url.search}${url.hash}`);
              demoModeRef.current = false;
              setDemoScenario(null);
            }}>Exit demo</button>
          </div>}
          <div className="page-heading">
            <div>
              <div className="eyebrow">LOCAL SYSTEM / THERMAL HEALTH</div>
              <h1>System overview<span className="heading-period">.</span></h1>
              <p>{warningDemo ? "Simulated warning preview; no hardware is being monitored." : "Live temperatures and a 10-minute thermal outlook for this PC."}</p>
            </div>
            <div className={`monitoring-badge ${status?.active ? "live" : "offline"}`}>
              <span className="pulse-dot" />
              {warningDemo ? "DEMO SIMULATION" : status?.active ? "MONITORING ACTIVE" : "MONITORING INACTIVE"}
            </div>
          </div>

          {!warningDemo && <div className="dashboard-actions">
            <button className={`calibrate-button ${calibrationStatus?.collecting ? "stop-calibration" : ""}`}
              onClick={calibrationStatus?.collecting ? stopCalibration : startCalibration} disabled={calibrationStarting}>
              {calibrationStarting ? <RefreshCw size={15} className="spin" />
                : calibrationStatus?.collecting ? <Square size={14} /> : <Play size={15} />}
              {calibrationStarting ? "Updating calibration…"
                : calibrationStatus?.collecting ? "Stop calibration" : "Start calibration"}
            </button>
            <button className="preview-open" onClick={() => {
              const url = new URL(window.location.href);
              url.searchParams.set("demo", "normal");
              window.history.replaceState({}, "", `${url.pathname}${url.search}${url.hash}`);
              demoModeRef.current = true;
              setDemoScenario("normal");
            }}><TriangleAlert size={15} /> Preview warnings</button>
          </div>}
          {!warningDemo && calibrationMessage && <div className="calibration-message" role="status">
            <Thermometer size={16} /><span>{calibrationMessage}</span>
          </div>}
          {warningDemo && <section className="demo-controls" aria-label="Warning preview scenarios">
            <div><b>Preview scenario</b><span>Switch states instantly; all values are simulated.</span></div>
            {DEMO_SCENARIOS.map((scenario) => <button key={scenario.id}
              className={demoScenario === scenario.id ? "selected" : ""}
              aria-pressed={demoScenario === scenario.id}
              onClick={() => {
                const url = new URL(window.location.href);
                url.searchParams.set("demo", scenario.id);
                window.history.replaceState({}, "", `${url.pathname}${url.search}${url.hash}`);
                demoModeRef.current = true;
                setDemoScenario(scenario.id);
              }}>{scenario.label}</button>)}
          </section>}

          {error && <div className="error-banner"><TriangleAlert size={17} /><span>{error}</span><button onClick={refresh}>Retry</button></div>}

          {activeAlerts.length > 0 && (
            <section className={`alert-banner ${warningDemo ? `demo-alert ${activeAlerts[0].severity.toLowerCase()}` : ""}`} aria-live="polite">
              <div className="alert-banner-icon"><AlertTriangle size={19} /></div>
              <div className="alert-banner-copy">
                <b>{activeAlerts[0].message}</b>
                <span>{activeAlerts[0].suggestedAction}</span>
              </div>
              <span className={`severity ${activeAlerts[0].severity.toLowerCase()}`}>{activeAlerts[0].severity}</span>
            </section>
          )}

          <div className="section-heading">
            <h2>Current conditions</h2><span className="updated"><i /> Updated {formatTime(current?.timestamp)}</span>
          </div>
          {loading && !current ? (
            <div className="empty-panel"><RefreshCw className="spin" size={20} /> Connecting to local sensor feed…</div>
          ) : !status?.active ? (
            <div className={`empty-panel disconnected-panel ${warningDemo ? "demo-unavailable" : ""}`}>
              <Radio size={22} /><div><b>Hardware sensor feed unavailable</b>
                <span>{warningDemo ? "This preview represents a sensor outage. No real sensor request was made." : status?.message || "The backend starts the hardware collector automatically; check collector startup logs."}</span></div>
            </div>
          ) : (
            <div className="stats-grid">
              <StatCard title="CPU temperature" value={formatTemp(current?.cpuTemperatureC)}
                subtitle={`Warning threshold ${cpuThreshold}°C`} icon={Cpu}
                status={cpuState} accent="#72e3bd" />
              <StatCard title="GPU temperature" value={formatTemp(current?.gpuTemperatureC)}
                subtitle={`Warning threshold ${gpuThreshold}°C`} icon={Zap}
                status={gpuState} accent="#8f9cff" />
              <StatCard title="CPU utilization"
                value={current?.cpuUtilizationPercent == null ? "--" : `${Number(current.cpuUtilizationPercent).toFixed(0)}%`}
                subtitle="Current processor load" icon={Activity} accent="#f6bd74" />
              <StatCard title="CPU clock speed"
                value={current?.cpuClockMhz == null ? "--" : `${(current.cpuClockMhz / 1000).toFixed(2)} GHz`}
                subtitle="Highest reported core clock" icon={Gauge} accent="#67c5e8" />
              <StatCard title="CPU power"
                value={current?.cpuPowerW == null ? "--" : `${Number(current.cpuPowerW).toFixed(1)} W`}
                subtitle="Reported processor package power" icon={Zap} accent="#f6bd74"
                status={current?.cpuPowerW == null ? "unavailable" : "normal"} />
              <StatCard title="CPU fan speed"
                value={current?.cpuFanSpeedRpm == null ? "--" : `${Math.round(current.cpuFanSpeedRpm)} RPM`}
                subtitle="Reported CPU fan sensor" icon={Fan} accent="#8f9cff"
                status={current?.cpuFanSpeedRpm == null ? "unavailable" : "normal"} />
              <StatCard title="RAM usage"
                value={current?.memoryUtilizationPercent == null ? "--" : `${Number(current.memoryUtilizationPercent).toFixed(0)}%`}
                subtitle="Reported memory load" icon={HardDrive} accent="#67c5e8"
                status={current?.memoryUtilizationPercent == null ? "unavailable" : "normal"} />
            </div>
          )}

          {!status?.active && !warningDemo && <div className="stats-grid temperature-offline">
            <StatCard title="CPU temperature" value="Unavailable" subtitle="No fresh hardware reading" icon={Cpu} status="unavailable" accent="#72e3bd" />
            <StatCard title="GPU temperature" value="Unavailable" subtitle="No fresh hardware reading" icon={Zap} status="unavailable" accent="#8f9cff" />
          </div>}

          <section className="panel process-panel" id="apps">
            <div className="panel-heading">
              <div><div className="eyebrow">LIVE PROCESS MONITOR</div><h2>Apps using CPU</h2></div>
              <span className="history-range">{warningDemo ? "Simulated" : processUsage?.available ? "Live · 5 sec samples" : "Waiting for collector"}</span>
            </div>
            <p className="process-note">CPU percentage is each app’s share of total processor capacity. Ending an app may discard unsaved work.</p>
            {processUsage?.available && processUsage.processes?.length ? <div className="process-list">
              {processUsage.processes.map((process) => <div className="process-row" key={process.processId}>
                <div className="process-title"><b>{process.processName}</b><span>PID {process.processId}{process.foreground ? " · In use" : ""}</span></div>
                <div className="process-meter"><i style={{ width: `${Math.min(100, process.cpuUsagePercent)}%` }} /></div>
                <b className="process-cpu">{Number(process.cpuUsagePercent).toFixed(1)}%</b>
                <span className="process-memory">{Number(process.memoryMb).toFixed(0)} MB RAM</span>
                <button className="end-app-button" disabled={warningDemo || endingPid === process.processId}
                  onClick={() => endApp(process)}>{endingPid === process.processId ? "Closing…" : "End app"}</button>
              </div>)}
            </div> : <div className="inline-empty"><Activity size={17} />{processUsage?.message || "Waiting for process CPU samples."}</div>}
          </section>

          <div className="content-grid">
            <section className="panel chart-panel" id="history">
              <div className="panel-heading">
                <div><div className="eyebrow">SENSOR HISTORY</div><h2>Temperature trend</h2></div>
                <span className="history-range">Recent readings <ChevronDown size={14} /></span>
              </div>
              {hasTemperatureHistory ? (
                <div className="chart-wrap">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={chartData} margin={{ top: 12, right: 6, left: -22, bottom: 0 }}>
                      <defs>
                        <linearGradient id="cpuFill" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#6fe1b8" stopOpacity={0.22} />
                          <stop offset="100%" stopColor="#6fe1b8" stopOpacity={0} />
                        </linearGradient>
                        <linearGradient id="gpuFill" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#929dff" stopOpacity={0.18} />
                          <stop offset="100%" stopColor="#929dff" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid stroke="#202a36" vertical={false} />
                      <XAxis dataKey="time" tick={{ fill: "#778697", fontSize: 11 }} axisLine={false} tickLine={false} minTickGap={28} />
                      <YAxis domain={["dataMin - 5", "dataMax + 5"]} tick={{ fill: "#778697", fontSize: 11 }} axisLine={false} tickLine={false} />
                      <Tooltip contentStyle={{ background: "#111a24", border: "1px solid #283544", borderRadius: 9 }}
                        labelStyle={{ color: "#d9e3eb" }} formatter={(value) => [`${Number(value).toFixed(1)}°C`]} />
                      <Area type="monotone" dataKey="cpu" name="CPU" stroke="#72e3bd" strokeWidth={2} fill="url(#cpuFill)" connectNulls />
                      <Area type="monotone" dataKey="gpu" name="GPU" stroke="#929dff" strokeWidth={2} fill="url(#gpuFill)" connectNulls />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <div className="chart-empty"><HardDrive size={20} /><span>{status?.active ? "More readings are needed to draw a trend." : "Sensor history appears here when the collector is running."}</span></div>
              )}
              <div className="chart-legend"><span><i className="cpu-dot" /> CPU</span><span><i className="gpu-dot" /> GPU</span>
                <span className="chart-unit">Temperature · °C</span></div>
            </section>

            <section className="panel forecast-panel">
              <div className="panel-heading">
                <div><div className="eyebrow">PREDICTIVE OUTLOOK</div><h2>In 10 minutes</h2></div>
                <span className="estimate-tag"><Zap size={13} />
                  CPU {warningDemo ? "DEMO" : cpuModelReady ? "READY" : "NOT READY"} · GPU {warningDemo ? "DEMO" : gpuModelReady ? "READY" : "NOT READY"}
                </span>
              </div>
              <p className="forecast-note">{predictionLabel}</p>
              <div className="forecast-list">
                {[["CPU", cpuForecast], ["GPU", gpuForecast]].map(([component, forecast]) => {
                  const threshold = component === "CPU" ? cpuThreshold : gpuThreshold;
                  const state = statusFor(forecast?.predictedC, threshold);
                  return <div className="forecast-row" key={component}>
                    <div className={`forecast-symbol ${component.toLowerCase()}`}>{component === "CPU" ? <Cpu size={17} /> : <Zap size={17} />}</div>
                    <div className="forecast-component"><b>{component}</b><span>Current {formatTemp(forecast?.currentC)} · threshold {threshold}°C</span></div>
                    <div className="forecast-number"><b className={state}>{forecast?.predictedC == null ? "--" : Number(forecast.predictedC).toFixed(1)}<small>°C</small></b>
                      <span>{warningDemo ? `Demo value · ${state}` : forecast?.status === "READY" ? `${forecastReadiness(forecast)} · ${state}` : forecastReadiness(forecast)}</span></div>
                  </div>;
                })}
              </div>
              <div className="model-note"><div className="model-indicator"><Activity size={14} /></div>
                <div><b>{warningDemo ? "DEMO FORECAST · simulated" : cpuModelReady ? (cpuForecast?.model || prediction?.model || "Trained model") : "CPU model not ready"}</b><span>
                  {warningDemo ? "Illustrative value only; no trained model or live sensor prediction is used." : cpuModelReady
                    ? "Forecast from a locally trained model. See training report for held-out metrics."
                    : cpuForecast?.model || "No CPU prediction is available until validated model inference is ready."}
                </span></div>
              </div>
              <div className="target-time"><span>Prediction generated</span><b>{formatTime(prediction?.generatedAt)}</b></div>
              <div className="target-time"><span>Forecast target</span><b>{formatTime(prediction?.targetTime)}</b></div>
            </section>
          </div>

          <div className="content-grid lower-grid">
            <section className="panel alerts-panel" id="alerts">
              <div className="panel-heading">
                <div><div className="eyebrow">EVENT LOG</div><h2>Recent alerts</h2></div>
                <span className="alert-total">{alerts.length} events</span>
              </div>
              {alerts.length ? <div className="alert-list">
                {alerts.slice(0, 5).map((alert) => <div className="alert-item" key={alert.id}>
                  <span className={`alert-icon ${alert.severity.toLowerCase()}`}><AlertTriangle size={15} /></span>
                  <div className="alert-description"><b>{alert.message}</b><span>{alert.suggestedAction}</span></div>
                  <time>{formatTime(alert.createdAt)}</time>
                  <span className={`alert-state ${alert.active ? "active" : ""}`}>{alert.active ? "ACTIVE" : "RESOLVED"}</span>
                </div>)}
              </div> : <div className="inline-empty"><ShieldCheck size={17} /> No thermal alerts recorded.</div>}
            </section>

            <section className="panel settings-panel">
              <div className="panel-heading">
                <div><div className="eyebrow">PREFERENCES</div><h2>Warning settings</h2></div>
                <Settings2 size={17} className="heading-icon" />
              </div>
              {settings && <div className="settings-form">
                <label className="threshold-field"><span>CPU warning threshold</span>
                  <div className="number-input"><input type="number" min="50" max="110" step="1" value={settings.cpuWarningC}
                    onChange={(event) => setSettings({ ...settings, cpuWarningC: Number(event.target.value) })}
                    onBlur={() => updateSettings({ cpuWarningC: settings.cpuWarningC })} /><small>°C</small></div>
                </label>
                <label className="threshold-field"><span>GPU warning threshold</span>
                  <div className="number-input"><input type="number" min="50" max="110" step="1" value={settings.gpuWarningC}
                    onChange={(event) => setSettings({ ...settings, gpuWarningC: Number(event.target.value) })}
                    onBlur={() => updateSettings({ gpuWarningC: settings.gpuWarningC })} /><small>°C</small></div>
                </label>
                <div className="setting-divider" />
                <div className="notification-setting">
                  <div className="notification-copy"><b>Browser notifications</b>
                    <span>{warningDemo ? "Muted during warning previews" : permission === "granted" ? "Permission granted" : permission === "denied" ? "Permission denied in browser" : "In-app alerts always remain available"}</span></div>
                  <button className={`toggle ${settings.notificationsEnabled ? "on" : ""}`}
                    aria-label="Toggle browser notifications" role="switch" aria-checked={settings.notificationsEnabled}
                    disabled={warningDemo}
                    onClick={() => updateSettings({ notificationsEnabled: !settings.notificationsEnabled })}>
                    <span />
                  </button>
                </div>
                {!warningDemo && settings.notificationsEnabled && permission !== "granted" && (
                  <button className="permission-button" onClick={requestNotificationPermission}>
                    {permission === "denied" ? <BellOff size={14} /> : <Bell size={14} />}
                    {permission === "denied" ? "Enable in browser settings" : "Allow desktop notifications"}
                  </button>
                )}
                {saving && <span className="saving-note"><RefreshCw size={12} className="spin" /> Saving settings…</span>}
                {settings && !saving && !error && <span className="saved-note"><Check size={12} />{warningDemo ? " Demo settings only" : " Settings saved locally"}</span>}
              </div>}
            </section>
          </div>

          <footer className="footer">
            <span><Fan size={13} /> No fan or hardware controls are changed by SmartTherm.</span>
            <span>Sensor update: {formatTime(current?.timestamp)} <span className="footer-separator">·</span> 5 sec sampling</span>
          </footer>
        </div>
      </section>
    </main>
  );
}

export default App;
