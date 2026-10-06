#!/usr/bin/env python3
"""Driver for the CPC-Hypergraph ablation ladder (spec 6.2).

For every (cell, seed) pair it clones the cell's yaml config with a unique
``log_name``/``model_file``, shells out to ``run_crslab.py``, then parses the
resulting log for the final rec/conv test metrics and the learned
scope-fusion weights (alpha_f). Results are appended to a JSONL file (safe to
resume/rerun) and aggregated into markdown summary tables.

Usage:
    uv run scripts/run_ablation.py run --seeds 3407                 # 1 seed, every cell in LADDER
    uv run scripts/run_ablation.py run --seeds 3407 42 123           # spec-mandated x3
    uv run scripts/run_ablation.py run --cells A0 A1 A2 --seeds 3407 # subset
    uv run scripts/run_ablation.py summarize                        # rebuild tables only
    uv run scripts/run_ablation.py run --cells A9 A9_v22_sep --seeds 3407   # ReDial v2.1 vs v2.2-sep
    uv run scripts/run_ablation.py run --dataset tgredial --seeds 123 --gpu-profile rtx6000   # full HTGReDial ladder (T*)
    uv run scripts/run_ablation.py run --dataset tgredial --cells T9 T9_norev --seeds 3407    # review on/off
    uv run scripts/run_ablation.py summarize --dataset tgredial

Two cells from the spec's table (docs/contexual_personal_collective/
cpc_hypergraph_v2.1_method_spec.md §6.2) are BLOCKED and skipped by default:
  A4  - needs the old MHIM retrieval-extension baseline. `extension_strategy`
        is read by hycorec.py but never used anywhere else -- the mechanism
        was dropped in the CPC-Hypergraph refactor.
  A11 - needs HyCoRec's review Transformer P_r. hycorec.py's own comment
        ("minus the review Transformer P_r which this codebase lacks") says
        it was never ported, so there is nothing to "turn off".
Both need real implementation work before they can be run; see --cells to
override once that lands.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent  # .../HyCoRec
ABLATION_CFG_DIR = ROOT / "config" / "crs" / "hycorec" / "ablation"
RUN_CFG_DIR = ROOT / "config" / "crs" / "hycorec" / "ablation" / "_runs"
LOG_DIR = ROOT / "log"
RESULTS_DIR = ROOT / "results" / "ablation"  # ReDial; other datasets: results/ablation_<dataset>/

# Datasets: each has its own config subdir, cell ladder and results dir (so
# summaries never mix datasets). "hredial" keeps the original layout.
DATASETS = ("hredial", "tgredial")
DATASET_CFG_DIRS = {"hredial": ABLATION_CFG_DIR, "tgredial": ABLATION_CFG_DIR / "tgredial"}

# GPU profiles: same cell->filename ladder, resolved under a different config
# subdir with a different batch_size per hardware (see --gpu-profile).
GPU_PROFILE_SUBDIRS = {
    "default": "",
    "rtx6000": "rtx6000",  # 16GB cards (V100/P100): same configs as default, batch_size x2
}
GPU_PROFILES = list(GPU_PROFILE_SUBDIRS)


def cfg_dir_for(dataset: str, profile: str) -> Path:
    base = DATASET_CFG_DIRS[dataset]
    sub = GPU_PROFILE_SUBDIRS[profile]
    return base / sub if sub else base


def results_dir_for(dataset: str) -> Path:
    return RESULTS_DIR if dataset == "hredial" else RESULTS_DIR.parent / f"ablation_{dataset}"

# Full ladder: every config under ablation/ (and each GPU profile subdir) runs by default.
#   A0-A9          spec ladder (A4/A11 excluded -- see BLOCKED)
#   A3b/A9b1       non-spec debug configs
#   A9_v22*        ReDial CPC v2.2 variants (need data/collective/hredial_unified_full)
# HTGReDial cells live in TG_LADDER (config/.../ablation/tgredial/, see its README.md).
# Results of different datasets never mix: each dataset has its own results dir.
LADDER = {
    "A0": "A0.yaml",
    "A1": "A1.yaml",
    "A2": "A2.yaml",
    "A3": "A3.yaml",
    "A3b": "A3b.yaml",
    "A5": "A5.yaml",
    "A6": "A6.yaml",
    "A7": "A7.yaml",
    "A8": "A8.yaml",
    "A9": "A9.yaml",
    "A9b1": "A9b1.yaml",
    "A9_v22": "A9_v22.yaml",
    "A9_v22_noword": "A9_v22_noword.yaml",
    "A9_v22_sep": "A9_v22_sep.yaml",
}
# HTGReDial ladder (mirrors the ReDial A-ladder; T = TG-ReDial).
TG_LADDER = {c: f"{c}.yaml" for c in [
    "T0", "T1", "T2", "T3", "T3b", "T5", "T6", "T7", "T8",
    "T9", "T9_norev", "T9_w2", "T9_w5",
]}
LADDERS = {"hredial": LADDER, "tgredial": TG_LADDER}
BLOCKED = {
    "A4": "needs the old MHIM retrieval-extension (extension_strategy is dead code)",
    "A11": "needs HyCoRec's review Transformer P_r (never implemented in this codebase)",
}

REC_METRIC_KEYS = [
    "recall@1", "recall@10", "recall@50",
    "mrr@1", "mrr@10", "mrr@50",
    "ndcg@1", "ndcg@10", "ndcg@50",
    "tail_recall@1", "tail_recall@10", "tail_recall@50",
]
CONV_METRIC_KEYS = ["bleu@1", "bleu@2", "bleu@3", "bleu@4", "dist@1", "dist@2", "dist@3", "dist@4", "dist_cnt@2", "dist_cnt@3", "dist_cnt@4", "f1"]

TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
REPORT_RE = re.compile(r"\| (?:crslab\.evaluator\.standard:report|.*:report):\d+ - \s*$")
STAGE_RE = re.compile(r"\[(Recommendation|Conversation) epoch \d+\]")
MODE_RE = re.compile(r"- \[(Train|Valid|Test)\]\s*$")
ALPHA_RE = re.compile(r"\[Scope fusion weights alpha_f\] (\{.*\})\s*$")


def run_id(cell: str, seed: int, profile: str = "default") -> str:
    suffix = "" if profile == "default" else f"_{profile}"
    return f"ablation_{cell}_seed{seed}{suffix}"


def new_stamp() -> str:
    """Timestamp naming one launch of this script: logs go to log/<stamp>/."""
    return time.strftime("%Y%m%d-%H%M%S")


def build_run_config(cell: str, cfg_file: str, seed: int, log_name: str | None = None, cfg_dir: Path = ABLATION_CFG_DIR, stamp: str | None = None) -> Path:
    """Clone the cell's yaml with a unique log_name/model_file so parallel/serial
    reruns never clobber each other's logs or checkpoints. With ``stamp`` the log
    goes to log/<stamp>/<rid>.log and the checkpoint to <rid>_<stamp>.pth."""
    src = cfg_dir / cfg_file
    opt = yaml.safe_load(src.read_text())
    rid = log_name or run_id(cell, seed)
    opt["log_name"] = f"{stamp}/{rid}" if stamp else rid
    opt["model_file"] = f"{rid}_{stamp}.pth" if stamp else f"{rid}.pth"
    run_cfg_dir = RUN_CFG_DIR / stamp if stamp else RUN_CFG_DIR
    run_cfg_dir.mkdir(parents=True, exist_ok=True)
    dst = run_cfg_dir / f"{rid}.yaml"
    dst.write_text(yaml.safe_dump(opt, sort_keys=False))
    return dst


def parse_log(log_path: Path) -> dict:
    """Extract final rec/conv test metrics + alpha_f table from a run's log file."""
    out = {
        "rec_test": None, "conv_test": None, "rec_valid": None, "conv_valid": None,
        "alpha": None,
        "epochs_rec": 0, "epochs_conv": 0, "early_stopped": False,
        "reached_conv_test": False, "wall_seconds": None,
    }
    if not log_path.exists():
        return out

    lines = log_path.read_text(errors="replace").splitlines()
    stage = None
    mode = None
    last_rec_valid = last_conv_valid = None
    first_ts = last_ts = None
    i = 0
    while i < len(lines):
        line = lines[i]
        m = TS_RE.match(line)
        if m:
            last_ts = m.group(1)
            if first_ts is None:
                first_ts = last_ts
        if STAGE_RE.search(line):
            stage = "rec" if "Recommendation" in line else "conv"
            if stage == "rec" and "[Recommendation epoch" in line:
                out["epochs_rec"] += 1
            if stage == "conv" and "[Conversation epoch" in line:
                out["epochs_conv"] += 1
        mm = MODE_RE.search(line)
        if mm:
            mode = mm.group(1).lower()
        if "[Early stop]" in line:
            out["early_stopped"] = True
        am = ALPHA_RE.search(line)
        if am:
            try:
                out["alpha"] = ast.literal_eval(am.group(1))
            except (ValueError, SyntaxError):
                pass
        if REPORT_RE.search(line) and i + 1 < len(lines):
            payload = lines[i + 1].strip()
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                data = None
            if data is not None and stage and mode:
                if stage == "rec" and mode == "valid":
                    last_rec_valid = data
                elif stage == "rec" and mode == "test":
                    out["rec_test"] = data
                    out["rec_valid"] = last_rec_valid
                elif stage == "conv" and mode == "valid":
                    last_conv_valid = data
                elif stage == "conv" and mode == "test":
                    out["conv_test"] = data
                    out["conv_valid"] = last_conv_valid
                    out["reached_conv_test"] = True
            i += 1
        i += 1

    if first_ts and last_ts:
        fmt = "%Y-%m-%d %H:%M:%S"
        out["wall_seconds"] = time.mktime(time.strptime(last_ts, fmt)) - time.mktime(
            time.strptime(first_ts, fmt)
        )
    return out


def run_one(cell: str, cfg_file: str, seed: int, gpu: str, timeout_hours: float | None, debug: bool = False, cfg_dir: Path = ABLATION_CFG_DIR, profile: str = "default", stamp: str | None = None, ddp: bool = False) -> dict:
    rid = run_id(cell, seed, profile)
    if debug:
        rid += "_debug"
    stamp = stamp or new_stamp()
    run_cfg = build_run_config(cell, cfg_file, seed, log_name=rid, cfg_dir=cfg_dir, stamp=stamp)
    log_dir = LOG_DIR / stamp
    stderr_path = log_dir / f"{rid}.stderr"
    log_path = log_dir / f"{rid}.log"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"==> [{rid}] launching (config={run_cfg.relative_to(ROOT)}, gpu={gpu}, debug={debug})", flush=True)
    t0 = time.time()
    runner = ["uv", "run"] if shutil.which("uv") else [sys.executable]
    cmd = runner + ["run_crslab.py", "-c", str(run_cfg.relative_to(ROOT)), "-g", gpu, "-s", str(seed)]
    n_gpu = len(gpu.split(","))
    if ddp and gpu != "-1" and n_gpu > 1:
        # one process per GPU; --standalone picks a free rendezvous port
        py = ["uv", "run", "python"] if shutil.which("uv") else [sys.executable]
        cmd = py + ["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={n_gpu}"] + cmd[len(runner):]
    if debug:
        cmd.append("-d")
    result = {"cell": cell, "seed": seed, "profile": profile, "stamp": stamp, "config": cfg_file, "run_id": rid, "cmd": " ".join(cmd)}
    try:
        with open(stderr_path, "w") as errf:
            # own process group: on timeout kill the whole tree, not just the
            # `uv run` wrapper (otherwise run_crslab.py keeps training orphaned).
            proc = subprocess.Popen(
                cmd, cwd=ROOT, stderr=errf, stdout=subprocess.DEVNULL, start_new_session=True,
            )
            try:
                proc.wait(timeout=timeout_hours * 3600 if timeout_hours else None)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                raise
        result["returncode"] = proc.returncode
    except subprocess.TimeoutExpired:
        result["returncode"] = None
        result["error"] = f"timed out after {timeout_hours}h"
    result["launch_wall_seconds"] = time.time() - t0

    parsed = parse_log(log_path)
    result.update(parsed)
    result["log_path"] = str(log_path.relative_to(ROOT))
    if result.get("returncode") not in (0, None):
        result.setdefault("error", f"run_crslab.py exited {result['returncode']}; see {stderr_path.relative_to(ROOT)}")
    elif result.get("returncode") is None and "error" not in result:
        result["error"] = "unknown failure"
    elif not parsed["reached_conv_test"]:
        result["error"] = "did not reach conversation test stage (see log/stderr)"

    status = "OK" if not result.get("error") else f"FAILED: {result['error']}"
    print(f"<== [{rid}] {status} ({result['launch_wall_seconds']/60:.1f} min)", flush=True)
    return result


def already_done(results_path: Path, cell: str, seed: int, profile: str = "default") -> bool:
    if not results_path.exists():
        return False
    for line in results_path.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            r.get("cell") == cell
            and r.get("seed") == seed
            and r.get("profile", "default") == profile
            and not r.get("error")
        ):
            return True
    return False


def cmd_run(args: argparse.Namespace) -> None:
    ladder = LADDERS[args.dataset]
    cells = {}
    for name in args.cells or list(ladder):
        if name in ladder:
            cells[name] = ladder[name]
        elif name in BLOCKED:
            print(f"!! skipping {name}: BLOCKED ({BLOCKED[name]})")
        else:
            print(f"!! unknown cell {name!r}, skipping")

    if args.gpu_profile not in GPU_PROFILES:
        print(f"!! unknown --gpu-profile {args.gpu_profile!r}, choices: {list(GPU_PROFILES)}")
        sys.exit(1)
    cfg_dir = cfg_dir_for(args.dataset, args.gpu_profile)

    missing = {c: f for c, f in cells.items() if not (cfg_dir / f).is_file()}
    for c, f in missing.items():
        print(f"!! skipping {c}: {cfg_dir.relative_to(ROOT) / f} does not exist for --gpu-profile {args.gpu_profile}")
    cells = {c: f for c, f in cells.items() if c not in missing}

    results_dir = results_dir_for(args.dataset)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "results.jsonl"
    stamp = new_stamp()
    print(f"Logs: log/{stamp}/")

    plan = [(c, f, s) for c, f in cells.items() for s in args.seeds]
    print(f"Ablation plan: {len(plan)} runs ({len(cells)} cells x {len(args.seeds)} seeds, gpu-profile={args.gpu_profile})")
    for cell, _, seed in plan:
        skip = (not args.force) and already_done(results_path, cell, seed, args.gpu_profile)
        print(f"  - {cell} seed={seed}" + ("  [skip: already done]" if skip else ""))

    for cell, cfg_file, seed in plan:
        if (not args.force) and already_done(results_path, cell, seed, args.gpu_profile):
            continue
        result = run_one(cell, cfg_file, seed, args.gpu, args.timeout_hours, cfg_dir=cfg_dir, profile=args.gpu_profile, stamp=stamp, ddp=args.ddp)
        with open(results_path, "a") as f:
            f.write(json.dumps(result) + "\n")
        if result.get("error") and args.stop_on_error:
            print(f"Stopping: {cell} seed={seed} failed and --stop-on-error was set.")
            sys.exit(1)

    build_summary(results_path, dataset=args.dataset)
    build_val_summary(results_path, dataset=args.dataset)


def mean_std(vals: list[float]) -> str:
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    if len(vals) == 1:
        return f"{vals[0]:.4f}"
    return f"{statistics.mean(vals):.4f}±{statistics.pstdev(vals):.4f}"


def build_summary(results_path: Path, profile_filter: str | None = None, dataset: str = "hredial") -> None:
    if not results_path.exists():
        print("No results yet.")
        return
    rows = [json.loads(l) for l in results_path.read_text().splitlines() if l.strip()]
    if profile_filter is not None:
        rows = [r for r in rows if r.get("profile", "default") == profile_filter]

    def label(r: dict) -> str:
        profile = r.get("profile", "default")
        return r["cell"] if profile == "default" else f"{r['cell']} [{profile}]"

    by_cell: dict[str, list[dict]] = {}
    for r in rows:
        # keep partial rows (e.g. rec finished, conv got interrupted) -- only
        # drop rows with no usable metrics at all.
        if not r.get("rec_test") and not r.get("conv_test"):
            continue
        by_cell.setdefault(label(r), []).append(r)

    order = list(LADDERS[dataset])
    cells_present = [c for c in order if c in by_cell] + [c for c in by_cell if c not in order]

    rec_lines = ["| cell | n | " + " | ".join(REC_METRIC_KEYS) + " |",
                 "|---" * (len(REC_METRIC_KEYS) + 2) + "|"]
    conv_lines = ["| cell | n | " + " | ".join(CONV_METRIC_KEYS) + " |",
                  "|---" * (len(CONV_METRIC_KEYS) + 2) + "|"]
    alpha_lines = ["| cell | seed | field | w_C | w_P | w_G |", "|---|---|---|---|---|---|"]

    for cell in cells_present:
        runs = by_cell[cell]
        rec_vals = {k: [r["rec_test"].get(k) for r in runs if r.get("rec_test")] for k in REC_METRIC_KEYS}
        conv_vals = {k: [r["conv_test"].get(k) for r in runs if r.get("conv_test")] for k in CONV_METRIC_KEYS}
        rec_lines.append(f"| {cell} | {len(runs)} | " + " | ".join(mean_std(rec_vals[k]) for k in REC_METRIC_KEYS) + " |")
        conv_lines.append(f"| {cell} | {len(runs)} | " + " | ".join(mean_std(conv_vals[k]) for k in CONV_METRIC_KEYS) + " |")
        for r in runs:
            if not r.get("alpha"):
                continue
            for field, w in r["alpha"].items():
                alpha_lines.append(f"| {cell} | {r['seed']} | {field} | {w[0]:.3f} | {w[1]:.3f} | {w[2]:.3f} |")

    failed = [r for r in rows if r.get("error")]
    fail_lines = ["| cell | seed | error |", "|---|---|---|"]
    for r in failed:
        fail_lines.append(f"| {label(r)} | {r['seed']} | {r['error']} |")

    out = results_dir_for(dataset) / "summary.md"
    out.write_text(
        "# Ablation results (spec 6.2)\n\n"
        f"Blocked cells (not runnable in current codebase): "
        + ", ".join(f"{k} ({v})" for k, v in BLOCKED.items()) + "\n\n"
        "## Recommendation metrics (test)\n\n" + "\n".join(rec_lines) + "\n\n"
        "## Conversation metrics (test)\n\n" + "\n".join(conv_lines) + "\n\n"
        "## Scope-fusion weights alpha_f (post rec-training, per field)\n\n" + "\n".join(alpha_lines) + "\n\n"
        + (("## Failed runs\n\n" + "\n".join(fail_lines) + "\n") if failed else "")
    )
    print(f"Wrote {out.relative_to(ROOT)} ({len(rows)} total runs, {len(failed)} failed)")


def build_val_summary(results_path: Path, profile_filter: str | None = None, dataset: str = "hredial") -> None:
    """Same tables as build_summary but sourced from the best-checkpoint valid
    reports (the [Valid] block right before the model is restored for [Test]),
    re-parsed from each run's log file -- results.jsonl predates rec_valid/
    conv_valid, so this never trusts stale jsonl fields for those keys."""
    if not results_path.exists():
        print("No results yet.")
        return
    rows = [json.loads(l) for l in results_path.read_text().splitlines() if l.strip()]
    if profile_filter is not None:
        rows = [r for r in rows if r.get("profile", "default") == profile_filter]

    def label(r: dict) -> str:
        profile = r.get("profile", "default")
        return r["cell"] if profile == "default" else f"{r['cell']} [{profile}]"

    by_cell: dict[str, list[dict]] = {}
    for r in rows:
        if not r.get("log_path"):
            continue
        reparsed = parse_log(ROOT / r["log_path"])
        if not reparsed["rec_valid"] and not reparsed["conv_valid"]:
            continue
        r = {**r, "rec_valid": reparsed["rec_valid"], "conv_valid": reparsed["conv_valid"]}
        by_cell.setdefault(label(r), []).append(r)

    order = list(LADDERS[dataset])
    cells_present = [c for c in order if c in by_cell] + [c for c in by_cell if c not in order]

    rec_lines = ["| cell | n | " + " | ".join(REC_METRIC_KEYS) + " |",
                 "|---" * (len(REC_METRIC_KEYS) + 2) + "|"]
    conv_lines = ["| cell | n | " + " | ".join(CONV_METRIC_KEYS) + " |",
                  "|---" * (len(CONV_METRIC_KEYS) + 2) + "|"]

    for cell in cells_present:
        runs = by_cell[cell]
        rec_vals = {k: [r["rec_valid"].get(k) for r in runs if r.get("rec_valid")] for k in REC_METRIC_KEYS}
        conv_vals = {k: [r["conv_valid"].get(k) for r in runs if r.get("conv_valid")] for k in CONV_METRIC_KEYS}
        rec_lines.append(f"| {cell} | {len(runs)} | " + " | ".join(mean_std(rec_vals[k]) for k in REC_METRIC_KEYS) + " |")
        conv_lines.append(f"| {cell} | {len(runs)} | " + " | ".join(mean_std(conv_vals[k]) for k in CONV_METRIC_KEYS) + " |")

    out = results_dir_for(dataset) / "summary_val.md"
    out.write_text(
        "# Ablation results (spec 6.2) -- validation split\n\n"
        "Same cells/runs as summary.md, but every metric is the best-checkpoint "
        "[Valid] report (the epoch restored just before [Test] runs), not the held-out test set. "
        "Use this to sanity-check test numbers against overfitting/selection noise, not as a substitute for summary.md.\n\n"
        f"Blocked cells (not runnable in current codebase): "
        + ", ".join(f"{k} ({v})" for k, v in BLOCKED.items()) + "\n\n"
        "## Recommendation metrics (valid)\n\n" + "\n".join(rec_lines) + "\n\n"
        "## Conversation metrics (valid)\n\n" + "\n".join(conv_lines) + "\n"
    )
    print(f"Wrote {out.relative_to(ROOT)} ({len(rows)} total runs)")


def cmd_summarize(args: argparse.Namespace) -> None:
    results_path = results_dir_for(args.dataset) / "results.jsonl"
    build_summary(results_path, profile_filter=args.gpu_profile, dataset=args.dataset)
    build_val_summary(results_path, profile_filter=args.gpu_profile, dataset=args.dataset)


def cmd_smoke(args: argparse.Namespace) -> None:
    """Fast harness check: trains on the (small) valid split via run_crslab.py -d.
    Does NOT write to results.jsonl / summary.md -- just proves the plumbing works."""
    cfg_file = LADDERS[args.dataset].get(args.cell)
    if not cfg_file:
        print(f"unknown cell {args.cell!r}")
        sys.exit(1)
    if args.gpu_profile not in GPU_PROFILES:
        print(f"!! unknown --gpu-profile {args.gpu_profile!r}, choices: {list(GPU_PROFILES)}")
        sys.exit(1)
    result = run_one(
        args.cell, cfg_file, args.seed, args.gpu, args.timeout_hours,
        debug=True, cfg_dir=cfg_dir_for(args.dataset, args.gpu_profile), profile=args.gpu_profile, ddp=args.ddp,
    )
    print(json.dumps(result, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def add_dataset(sp):
        sp.add_argument("--dataset", default="hredial", choices=DATASETS, help="which ablation ladder/config dir/results dir to use")

    p_run = sub.add_parser("run", help="run (missing) ablation cells and rebuild the summary")
    p_run.add_argument("--cells", nargs="*", default=None, help="cell names, default = every cell in LADDER (all configs; A4/A11 stay blocked)")
    p_run.add_argument("--seeds", nargs="+", type=int, default=[3407], help="spec asks for x3 seeds")
    p_run.add_argument("--gpu", default="0", help="GPU id string for run_crslab.py -g (use -1 for CPU)")
    p_run.add_argument("--ddp", action="store_true", help="with several --gpu ids, launch via torchrun (one process per GPU, DistributedDataParallel) instead of DataParallel")
    p_run.add_argument("--gpu-profile", default="default", choices=GPU_PROFILES,
                        help="config subdir to run from; 'rtx6000' = same ladder, batch_size x2 for 16GB GPUs")
    p_run.add_argument("--timeout-hours", type=float, default=None, help="kill a single run after N hours")
    p_run.add_argument("--force", action="store_true", help="rerun cells/seeds that already have a successful result")
    p_run.add_argument("--stop-on-error", action="store_true")
    add_dataset(p_run)
    p_run.set_defaults(func=cmd_run)

    p_sum = sub.add_parser("summarize", help="rebuild results/ablation/summary.md from results.jsonl without training")
    p_sum.add_argument("--gpu-profile", default=None, choices=GPU_PROFILES,
                        help="only summarize runs from this profile (default: all profiles)")
    add_dataset(p_sum)
    p_sum.set_defaults(func=cmd_summarize)

    p_smoke = sub.add_parser("smoke", help="fast harness check on the valid split (run_crslab.py -d); does not touch results.jsonl") 
    p_smoke.add_argument("--cell", default="A0")
    p_smoke.add_argument("--seed", type=int, default=3407)
    p_smoke.add_argument("--gpu", default="0")
    p_smoke.add_argument("--ddp", action="store_true", help="launch via torchrun, one process per GPU")
    p_smoke.add_argument("--gpu-profile", default="default", choices=GPU_PROFILES)
    p_smoke.add_argument("--timeout-hours", type=float, default=1.0)
    add_dataset(p_smoke)
    p_smoke.set_defaults(func=cmd_smoke)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
