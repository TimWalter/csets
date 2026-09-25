"""
Local stand-in for the CORA-COMP harness, plus a comparison against the public scoreboard.

    python benchmark/local_eval.py run [--benchmark B ...] [--filter REGEX] [--device cpu|gpu]
                                       [--mode all|first|random] [--catalog instances.csv]
    python benchmark/local_eval.py compare <results.csv or results dir> [--tools T ...]

`run` does what a platform worker does after installing the tool: for every selected row of
instances.csv it calls `prepare_instance.sh` (untimed) and then `run_instance.sh` with
`v1` + every catalog column in file order + a results file, measures the wall-clock time of
`run_instance.sh`, kills its process group at the row's `timeout`, and writes `results.csv`
and `run.log` per benchmark in the platform's format under `benchmark/local_results/<stamp>/`.

`compare` lines our times up against every tool on https://cpsvm5.cit.tum.de/api/cora/results/,
raw and with each tool's `test/startup-1d-cpu` overhead subtracted (the scoreboard's
"subtract overhead" toggle). It also works on a results directory downloaded from the platform.
"""
import argparse
import csv
import json
import os
import random
import re
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "benchmark" / ".cache"
CATALOG_URL = "https://raw.githubusercontent.com/CORA-COMP/benchmarks/main/instances.csv"
SCOREBOARD_URL = "https://cpsvm5.cit.tum.de/api/cora/results/data/"
OUR_TOOL = "csets (local)"
PREPARE_TIMEOUT = 600
STARTUP = ("test", "startup-1d-cpu")


def fetch(url: str, path: Path, refresh: bool) -> Path:
    if refresh or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as r:
            path.write_bytes(r.read())
    return path


def load_catalog(path: str | None, refresh: bool) -> tuple[list[str], list[dict]]:
    path = Path(path) if path else fetch(CATALOG_URL, CACHE / "instances.csv", refresh)
    with open(path, newline="") as f:
        rows = list(csv.reader(f, delimiter=";"))
    header = rows[0]
    return header, [dict(zip(header, r)) for r in rows[1:] if r]


def select(rows: list[dict], args) -> list[dict]:
    rows = [r for r in rows if r["benchmark"] in args.benchmark]
    if args.filter:
        rows = [r for r in rows if re.search(args.filter, r["instance"])]
    if args.device:
        rows = [r for r in rows if json.loads(r["params"])["device"] == args.device]
    by_bench = defaultdict(list)
    for r in rows:
        by_bench[r["benchmark"]].append(r)
    if args.mode == "first":
        by_bench = {b: rs[:1] for b, rs in by_bench.items()}
    elif args.mode == "random":
        rng = random.Random(args.seed)
        by_bench = {b: rng.sample(rs, min(10, len(rs))) for b, rs in by_bench.items()}
    # The platform runs one benchmark after another; `test` first so its startup is known early.
    return [r for b in sorted(by_bench, key=lambda b: (b != "test", b)) for r in by_bench[b]]


def call(cmd: list[str], timeout: float, log) -> tuple[str, float]:
    """Run `cmd` in its own process group; return ("ok"|"failed"|"timeout", wall seconds)."""
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True, text=True, errors="replace")
    try:
        out, _ = proc.communicate(timeout=timeout)
        status = "ok" if proc.returncode == 0 else "failed"
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        out, _ = proc.communicate()
        status = "timeout"
    elapsed = time.perf_counter() - t0
    for line in (out or "").splitlines():
        log.write(f"│ {line}\n")
    return status, elapsed


def read_result(path: str) -> dict:
    try:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        return rows[0] if rows else {}
    except OSError:
        return {}


def stop_daemon() -> None:
    """Kill the csets daemon prepare_instance.sh starts, so a run never reuses one with stale code
    and none is left holding GPU memory afterwards."""
    pid_file = Path(os.environ.get("CSETS_SERVER_DIR", REPO / ".server")) / "server.pid"
    try:
        pid = int(pid_file.read_text())
        if b"benchmark/server.py" in Path(f"/proc/{pid}/cmdline").read_bytes():
            os.kill(pid, signal.SIGKILL)
            print(f"stopped csets daemon {pid}")
    except (OSError, ValueError):
        pass
    pid_file.unlink(missing_ok=True)


def run(args) -> Path:
    stop_daemon()
    header, rows = load_catalog(args.catalog, args.refresh)
    rows = select(rows, args)
    if not rows:
        sys.exit("No instances match the selection.")
    out_dir = Path(args.out) if args.out else REPO / "benchmark" / "local_results" / datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"{len(rows)} instances -> {out_dir}")

    results = defaultdict(list)
    logs = {}
    for k, row in enumerate(rows, 1):
        bench, inst = row["benchmark"], row["instance"]
        (out_dir / bench).mkdir(parents=True, exist_ok=True)
        log = logs.setdefault(bench, open(out_dir / bench / "run.log", "a"))
        log.write(f"\n┏ Running instance {inst}\n")
        columns = [row[c] for c in header]
        timeout = float(row["timeout"]) if row.get("timeout") else None

        log.write("┌ prepare_instance.sh\n")
        prep_status, prep_time = call(["./prepare_instance.sh", "v1", *columns], PREPARE_TIMEOUT, log)
        log.write(f"└ prepare_instance.sh -> {prep_status} in {prep_time:.2f}s\n")
        record = {"benchmark": bench, "instance": inst, "prepare_time": round(prep_time, 4)}

        if prep_status != "ok":
            record.update(result="prepare_failed", time="")
        else:
            fd, result_file = tempfile.mkstemp(suffix=".csv")
            os.close(fd)
            log.write("┌ run_instance.sh\n")
            status, wall = call(["./run_instance.sh", "v1", *columns, result_file], timeout, log)
            tool = read_result(result_file)
            os.unlink(result_file)
            if status == "timeout":
                record.update(result="timeout", time=round(wall, 4))
            else:
                record.update({k: v for k, v in tool.items() if k != "result"})
                record.update(result=tool.get("result", "error"), time=round(wall, 4))
            log.write(f"└ run_instance.sh -> {record['result']} in {wall:.2f}s\n")
        log.flush()
        results[bench].append(record)
        extra = "  ".join(f"{c}={record[c]}" for c in record if c.startswith("time_"))
        print(f"[{k}/{len(rows)}] {bench}/{inst:28s} {record['result']:9s} {record['time']!s:>8}s  "
              f"(prepare {prep_time:.2f}s)  {extra}", flush=True)

    for bench, recs in results.items():
        tool_cols = sorted({c for r in recs for c in r} - {"benchmark", "instance", "prepare_time", "result", "time"})
        cols = ["benchmark", "instance", *tool_cols, "prepare_time", "result", "time"]
        with open(out_dir / bench / "results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, cols)
            w.writeheader()
            w.writerows(recs)
        logs[bench].close()
    if not args.keep_daemon:
        stop_daemon()
    print(f"\nResults in {out_dir}")
    return out_dir


# --- comparison -------------------------------------------------------------------------------

def load_ours(path: Path) -> dict:
    files = [path] if path.is_file() else sorted(path.glob("*/results.csv")) + sorted(path.glob("results.csv"))
    ours = {}
    for f in files:
        with open(f, newline="") as fh:
            for r in csv.DictReader(fh):
                ours[(r["benchmark"], r["instance"])] = (r["result"], float(r["time"]) if r["time"] else None)
    return ours


def compare(args) -> None:
    ours = load_ours(Path(args.results))
    board = json.loads(fetch(SCOREBOARD_URL, CACHE / "scoreboard.json", args.refresh).read_text())
    table = defaultdict(dict)  # tool -> (bench, inst) -> (result, time)
    for m in board["measurements"]:
        table[m["tool"].strip()][(m["benchmark"], m["instance"])] = (m["result"], m["time"])
    table[OUR_TOOL] = ours
    tools = [OUR_TOOL] + (args.tools or sorted(t for t in table if t != OUR_TOOL))

    def overhead(tool):
        r = table[tool].get(STARTUP)
        return r[1] if r and r[0] == "finished" else 0.0

    net = not args.raw
    print(f"Times in seconds{', minus each tool’s test/startup-1d-cpu time' if net else ''}. "
          f"Startup: " + ", ".join(f"{t} {overhead(t):.3f}" for t in tools))

    keys = sorted(k for k in ours if k != STARTUP)
    width = max(len(f"{b}/{i}") for b, i in keys) if keys else 20
    short = [t[:11] for t in tools]
    print(f"\n{'instance':{width}s} " + " ".join(f"{s:>11s}" for s in short) + "   ours/best")

    def cell(tool, key):
        r = table[tool].get(key)
        if not r:
            return None, "-"
        res, t = r
        if res != "finished":
            return None, res[:9]
        v = max(t - overhead(tool), 0.0) if net else t
        return v, f"{v:.3f}"

    ratios, wins, verdicts = [], Counter(), Counter()
    for key in keys:
        vals = [cell(t, key) for t in tools]
        others = [(v, t) for (v, _), t in zip(vals[1:], tools[1:]) if v is not None]
        best = min(others) if others else None
        mine = vals[0][0]
        verdicts[vals[0][1] if mine is None else "finished"] += 1
        ratio = ""
        if mine is not None and best:
            r = (mine + 1e-3) / (best[0] + 1e-3)
            ratios.append(r)
            ratio = f"{r:8.1f}x  (best: {best[1]})"
            wins[mine <= best[0]] += 1
        print(f"{key[0] + '/' + key[1]:{width}s} " + " ".join(f"{s:>11s}" for _, s in vals) + f"   {ratio}")

    if ratios:
        print(f"\n{len(keys)} instances; ours: {dict(verdicts)}. Fastest on {wins[True]}/{len(ratios)} "
              f"comparable instances; ours/best: median {statistics.median(ratios):.1f}x, "
              f"geo-mean {statistics.geometric_mean(ratios):.1f}x (1 ms added to both sides).")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run instances like the platform does")
    r.add_argument("--benchmark", action="append", help="benchmark(s) to run; default test and every zonotope and interval benchmark")
    r.add_argument("--filter", help="regex on the instance name, e.g. 'matMul-(10|1000)d'")
    r.add_argument("--device", choices=["cpu", "gpu"])
    r.add_argument("--mode", choices=["all", "first", "random"], default="all", help="as on the submission form")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--catalog", help="instances.csv to use; default: the current catalog from GitHub")
    r.add_argument("--out", help="output directory")
    r.add_argument("--no-compare", action="store_true", help="skip the scoreboard comparison at the end")
    r.add_argument("--keep-daemon", action="store_true", help="leave the csets daemon running afterwards")
    r.add_argument("--refresh", action="store_true", help="re-download catalog and scoreboard")

    c = sub.add_parser("compare", help="compare a results.csv / results directory against the scoreboard")
    c.add_argument("results")
    c.add_argument("--refresh", action="store_true")
    for sp in (r, c):
        sp.add_argument("--tools", nargs="+", help="scoreboard tools to show (default: all)")
        sp.add_argument("--raw", action="store_true", help="do not subtract each tool's startup overhead")

    args = p.parse_args()
    if args.cmd == "run":
        args.benchmark = args.benchmark or ["test", "zonotope", "zonotope-batched", "interval", "interval-batched"]
        out = run(args)
        if not args.no_compare:
            args.results = out
            compare(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
