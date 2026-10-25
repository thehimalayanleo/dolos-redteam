"""Log scalars for Run Tracker. Standard library only; copy this file anywhere.

    from runtracker import Logger
    log = Logger("~/RunTrackerLogs", "classifier-qwen27b", judge="classifier", model="Qwen3.8-27B-FP8")
    log.scalar("train/loss", 0.42, step=100)
    log.confusion("eval", "overall", tp=180, fp=12, tn=388, fn=20, step=100)

Layout (read by the app, which reloads every few seconds):
    <logdir>/<run>/scalars.jsonl   {"tag", "step", "value", "wall_time"} per line
    <logdir>/<run>/meta.json       optional judge / model / dataset / notes / source

A later value at the same (tag, step) replaces the earlier one in the app.
"""
import json
import math
import os
import re
import time
from pathlib import Path


def safe_name(name):
    return re.sub(r"[^\w.-]", "-", name)


class Logger:
    def __init__(self, logdir, run, **meta):
        self.dir = Path(os.path.expanduser(logdir)) / safe_name(run)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "scalars.jsonl"
        self.steps = {}
        if meta:
            (self.dir / "meta.json").write_text(json.dumps(meta, sort_keys=True))

    def scalar(self, tag, value, step=None):
        """Append one value. Without a step, each tag counts up from 0 on its own."""
        if step is None:
            step = self.steps.get(tag, -1) + 1
        self.steps[tag] = step
        row = {"tag": tag, "step": int(step), "value": float(value), "wall_time": time.time()}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    def confusion(self, prefix, axis, tp, fp, tn, fn, step=None, undecided=0):
        """Log precision, recall, FPR, F1, MCC, accuracy and coverage, plus the raw counts."""
        def ratio(a, b):
            return a / b if b else None
        p, r = ratio(tp, tp + fp), ratio(tp, tp + fn)
        d = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        decided = tp + fp + tn + fn
        rates = {"precision": p, "recall": r, "fpr": ratio(fp, fp + tn),
                 "f1": 2 * p * r / (p + r) if p is not None and r is not None and p + r > 0 else None,
                 "mcc": (tp * tn - fp * fn) / math.sqrt(d) if d else None,
                 "accuracy": ratio(tp + tn, decided), "coverage": ratio(decided, decided + undecided)}
        for name, value in rates.items():
            if value is not None:
                self.scalar(f"{prefix}/{axis}/{name}", value, step)
        for name, value in (("tp", tp), ("fp", fp), ("tn", tn), ("fn", fn), ("undecided", undecided)):
            self.scalar(f"counts/{prefix}/{axis}/{name}", value, step)


# ---- Tracked runs: one-line setup in the style of wandb.init ------------------------------------------------

def flatten(d, prefix=""):
    """{"opt": {"lr": 1e-3}} -> {"opt.lr": "0.001"}; values become strings so meta.json stays simple."""
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = v if isinstance(v, str) else json.dumps(v, default=str)
    return out


def _git():
    import subprocess

    def run(*args):
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    commit = run("rev-parse", "HEAD")
    if not commit:
        return {}
    return {"git_commit": commit[:12], "git_branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty": "yes" if run("status", "--porcelain") else "no", "git_root": run("rev-parse", "--show-toplevel")}


def _system_sample():
    """GPU and host load right now: nvidia-smi if present, else Apple GPU via ioreg; plus 1-minute load."""
    import subprocess
    vals = {}
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=power.draw,utilization.gpu,memory.used,temperature.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        for i, line in enumerate(out.strip().splitlines()):
            cells = [c.strip() for c in line.split(",")]
            for name, raw in zip(("gpu_power_w", "gpu_util_pct", "gpu_mem_used_gb", "gpu_temp_c"), cells):
                try:
                    v = float(raw)
                except ValueError:
                    continue
                vals[f"sys/{name}" + (f"/{i}" if i else "")] = v / 1024 if name == "gpu_mem_used_gb" else v
    except (OSError, subprocess.SubprocessError):
        try:
            out = subprocess.run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"],
                                 capture_output=True, text=True, timeout=10).stdout
            m = re.search(r'"Device Utilization %"=(\d+)', out)
            if m:
                vals["sys/gpu_util_pct"] = float(m.group(1))
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        vals["sys/load_1m"] = os.getloadavg()[0]
    except OSError:
        pass
    return vals


class Run:
    """A tracked run. Records config, git state, command and host in meta.json, logs metrics, optionally samples
    system metrics in the background, and marks itself finished or crashed.

        run = runtracker.init(project="bioscope · Training", name="esm29r-routed", config={"lr": 1e-3})
        run.log({"train/loss": 0.4, "eval/auc": 0.81}, step=100)
        run.save("model.pt")
        run.finish(summary={"test/auc": 0.79})
    Also a context manager: `with runtracker.init(...) as run:` marks the run crashed if the block raises.
    """

    def __init__(self, project=None, name=None, config=None, logdir="~/RunTrackerLogs", kind="training", title=None,
                 question=None, headline=None, system_metrics=True, interval=15.0, **meta):
        import socket
        import sys
        import threading
        name = name or f"run-{time.strftime('%Y%m%d-%H%M%S')}"
        self.logger = Logger(logdir, name)
        self.dir, self.step, self.started = self.logger.dir, 0, time.time()
        self.meta = {"project": project, "kind": kind, "title": title or name, "question": question,
                     "status": "running", "date": time.strftime("%Y-%m-%d %H:%M"), "compute": socket.gethostname(),
                     "config": flatten(config), "headline": headline, **meta}
        self.meta["extra"] = {**_git(), "command": " ".join(sys.argv), "host": socket.gethostname(),
                              "python": sys.version.split()[0], "pid": str(os.getpid()), **(meta.get("extra") or {})}
        self._write_meta()
        self._stop = threading.Event()
        self._thread = None
        if system_metrics:
            self._thread = threading.Thread(target=self._sample, args=(interval,), daemon=True)
            self._thread.start()
        import atexit
        atexit.register(self._atexit)

    def _write_meta(self):
        clean = {k: v for k, v in self.meta.items() if v not in (None, {}, [])}
        (self.dir / "meta.json").write_text(json.dumps(clean, sort_keys=True, indent=1))

    def _sample(self, interval):
        while not self._stop.wait(interval):
            for tag, v in _system_sample().items():
                self.logger.scalar(tag, v, step=int(time.time()))

    def log(self, metrics, step=None):
        """Log a dict of numbers. Without a step, each call advances the run's step by one."""
        if step is None:
            step = self.step
            self.step += 1
        else:
            self.step = max(self.step, int(step) + 1)
        for tag, v in metrics.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            if v == v and abs(v) != float("inf"):
                self.logger.scalar(tag, v, step=int(step))

    def config_update(self, config):
        self.meta["config"] = {**self.meta.get("config", {}), **flatten(config)}
        self._write_meta()

    def save(self, path):
        """Copy a file into the run's artifacts/ folder and record its size and SHA-256 in meta.json."""
        import hashlib
        import shutil
        src = Path(path).expanduser()
        dst = self.dir / "artifacts" / src.name
        dst.parent.mkdir(exist_ok=True)
        shutil.copy2(src, dst)
        digest = hashlib.sha256(dst.read_bytes()).hexdigest()[:16]
        self.meta["extra"][f"artifact {src.name}"] = f"{dst.stat().st_size} bytes, sha256 {digest}"
        self._write_meta()
        return dst

    def finish(self, summary=None, status="finished"):
        if self.meta["status"] != "running":
            return
        self._stop.set()
        if summary:
            self.log({f"summary/{k}" if "/" not in k else k: v for k, v in summary.items()}, step=max(self.step - 1, 0))
        self.meta["status"] = status
        self.meta["extra"]["duration"] = f"{(time.time() - self.started) / 60:.1f} min"
        self._write_meta()

    def _atexit(self):
        if self.meta["status"] == "running":
            self.finish(status="finished (at exit)")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.finish(status="crashed" if exc_type else "finished")
        return False


def init(project=None, name=None, config=None, **kwargs):
    """Start a tracked run; see Run."""
    return Run(project=project, name=name, config=config, **kwargs)
