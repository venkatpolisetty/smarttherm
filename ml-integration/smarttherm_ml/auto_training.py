from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .dataset import discover_csv_files
from .training import train_from_history

logger = logging.getLogger("smarttherm.ml.auto_training")


class BackgroundTrainingCoordinator:
    def __init__(
        self,
        project_root: Path,
        model_dir: Path,
        interval_seconds: int,
        enabled: bool = True,
    ) -> None:
        if interval_seconds < 60:
            raise ValueError("Automatic training interval must be at least 60 seconds.")
        self.project_root = project_root.resolve()
        self.model_dir = model_dir.resolve()
        self.interval_seconds = interval_seconds
        self.enabled = enabled
        self.state_file = self.model_dir / "auto-training-state.json"
        self._stop_event = threading.Event()
        self._run_lock = threading.Lock()
        self._status_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._last_attempted_fingerprint: str | None = None
        self._status: dict[str, Any] = {
            "enabled": enabled,
            "status": "NOT_CHECKED" if enabled else "DISABLED",
            "intervalSeconds": interval_seconds,
            "lastCheckedAt": None,
            "lastAttemptAt": None,
            "components": {},
        }
        self._load_state()

    def start(self) -> None:
        if not self.enabled:
            logger.info("Background model training is disabled.")
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="smarttherm-model-training",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=30)
            if thread.is_alive():
                logger.warning("Model training is still running during service shutdown.")

    def status(self) -> dict[str, Any]:
        with self._status_lock:
            return dict(self._status)

    def check_once(self) -> dict[str, Any]:
        if not self.enabled:
            return self.status()
        if not self._run_lock.acquire(blocking=False):
            logger.info("Skipping automatic training check because another training job is active.")
            return self.status()
        try:
            checked_at = datetime.now(timezone.utc).isoformat()
            fingerprint = self._fingerprint()
            if fingerprint == self._last_attempted_fingerprint:
                with self._status_lock:
                    self._status["lastCheckedAt"] = checked_at
                return self.status()

            with self._status_lock:
                self._status.update({
                    "status": "CHECKING",
                    "lastCheckedAt": checked_at,
                })
            report = train_from_history(self.project_root, self.model_dir)
            attempted_at = datetime.now(timezone.utc).isoformat()
            components = report["components"]
            status = report["status"]
            self._last_attempted_fingerprint = fingerprint
            self._write_state({
                "lastAttemptedFingerprint": (
                    fingerprint
                    if status not in {"TRAINING_ERROR", "TRAINING_BUSY"}
                    else self._last_attempted_fingerprint
                ),
                "lastCheckedAt": checked_at,
                "lastAttemptAt": attempted_at,
                "status": status,
                "components": components,
            })
            if status not in {"TRAINING_ERROR", "TRAINING_BUSY"}:
                self._last_attempted_fingerprint = fingerprint
            with self._status_lock:
                self._status.update({
                    "status": status,
                    "lastCheckedAt": checked_at,
                    "lastAttemptAt": attempted_at,
                    "components": components,
                    "validSensorRows": report["validSensorRows"],
                    "csvFilesInspected": report["csvFilesInspected"],
                })
                self._status.pop("error", None)
            logger.info(
                "Automatic training check finished: status=%s components=%s",
                status,
                {name: result["status"] for name, result in components.items()},
            )
            return self.status()
        except Exception as exception:
            logger.exception("Automatic model training check failed.")
            failed_at = datetime.now(timezone.utc).isoformat()
            with self._status_lock:
                self._status.update({
                    "status": "TRAINING_ERROR",
                    "lastCheckedAt": failed_at,
                    "error": str(exception),
                })
            return self.status()
        finally:
            self._run_lock.release()

    def request_check(self) -> dict[str, Any]:
        if not self.enabled:
            return self.status()
        with self._status_lock:
            if self._status.get("status") not in {"CHECKING", "QUEUED"}:
                self._status["status"] = "QUEUED"
        threading.Thread(
            target=self.check_once,
            name="smarttherm-model-training-request",
            daemon=True,
        ).start()
        return self.status()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self.check_once()
            if self._stop_event.wait(self.interval_seconds):
                return

    def _fingerprint(self) -> str:
        digest = hashlib.sha256()
        for path in discover_csv_files(self.project_root):
            stat = path.stat()
            digest.update(str(path).encode("utf-8"))
            digest.update(str(stat.st_size).encode("ascii"))
            digest.update(str(stat.st_mtime_ns).encode("ascii"))
        return digest.hexdigest()

    def _load_state(self) -> None:
        if not self.state_file.is_file():
            return
        try:
            state = json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exception:
            logger.warning("Could not read automatic training state %s: %s", self.state_file, exception)
            return
        if not isinstance(state, dict):
            logger.warning("Ignoring malformed automatic training state at %s.", self.state_file)
            return
        fingerprint = state.get("lastAttemptedFingerprint")
        if isinstance(fingerprint, str):
            self._last_attempted_fingerprint = fingerprint
        with self._status_lock:
            for key in ("status", "lastCheckedAt", "lastAttemptAt", "components"):
                if key in state:
                    self._status[key] = state[key]

    def _write_state(self, value: dict[str, Any]) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.model_dir,
                prefix=".auto-training-state.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                json.dump(value, handle, indent=2, allow_nan=False)
                handle.write("\n")
            os.replace(temporary, self.state_file)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
