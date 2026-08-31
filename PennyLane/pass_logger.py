"""
pass_logger.py — records what PennyLane transforms actually do.

PennyLane has no pass manager and no callback hook. Its optimizations are
*transforms*: plain functions from tape to tape.

    (new_tape,), _ = qml.transforms.cancel_inverses(tape)

So instead of hooking into a pipeline, this logger applies transforms one
at a time and measures around each call. That means the pipelines in
run_classical.py and run_quantum.py are OURS, not PennyLane's — it does
not ship preset optimization levels the way Qiskit does. Worth stating
plainly in any write-up.

Usage:

    import pass_logger
    tape = pass_logger.run_pipeline(tape, PIPELINE, case_id="my_case")
    pass_logger.summary()
    pass_logger.save("pass_log.csv")
"""

import csv
import time
import tracemalloc
from datetime import datetime

import pennylane as qml

# Every event recorded goes here.
EVENTS = []


def measure_tape(tape):
    """Measure a tape. Returns a dict of structural metrics."""
    blank = {
        "ops": -1, "depth": -1, "gates_2q": -1, "gates_1q": -1,
        "gate_types": -1, "wires": -1, "params": -1,
        "measurements": -1, "mid_measures": -1,
    }
    try:
        ops = tape.operations
    except Exception:
        return blank

    try:
        two_q = sum(1 for o in ops if len(o.wires) == 2)
        one_q = sum(1 for o in ops if len(o.wires) == 1)

        # Mid-circuit measurements and conditionals are their own op types.
        mid = sum(1 for o in ops
                  if type(o).__name__ in ("MidMeasure", "MidMeasureMP",
                                          "Conditional"))

        try:
            depth = tape.graph.get_depth()
        except Exception:
            depth = -1

        # Total trainable/positional parameters across all gates.
        try:
            n_params = sum(len(o.parameters) for o in ops)
        except Exception:
            n_params = -1

        return {
            "ops": len(ops),
            "depth": depth,
            "gates_2q": two_q,
            "gates_1q": one_q,
            "gate_types": len({o.name for o in ops}),
            "wires": len(tape.wires),
            "params": n_params,
            "measurements": len(tape.measurements),
            "mid_measures": mid,
        }
    except Exception:
        return blank


def apply_one(tape, name, transform, case_id, config, program, seq,
              **tf_kwargs):
    """Apply a single transform with measurement around it.

    Returns the new tape, or the original if the transform failed.
    """
    before = measure_tape(tape)

    tracemalloc.start()
    snap = tracemalloc.take_snapshot()
    t0 = time.perf_counter()
    failed = None
    try:
        (new_tape,), _ = transform(tape, **tf_kwargs)
    except Exception as e:
        failed = e
        new_tape = tape
    t1 = time.perf_counter()

    try:
        after_snap = tracemalloc.take_snapshot()
        _, peak = tracemalloc.get_traced_memory()
        mem = sum(s.size_diff
                  for s in after_snap.compare_to(snap, "filename"))
    except Exception:
        mem, peak = 0, 0
    tracemalloc.stop()

    after = measure_tape(new_tape)

    EVENTS.append({
        "case_id": case_id,
        "program": program,
        "config": config,
        "activity": name,
        "timestamp": datetime.fromtimestamp(t0).isoformat(),
        "seq": seq,

        "duration_ms": round((t1 - t0) * 1000, 4),
        "mem_alloc_kb": round(mem / 1024, 2),
        "mem_peak_kb": round(peak / 1024, 2),

        "changed": (before != after) and failed is None,
        "failed": failed is not None,

        "ops_before": before["ops"],
        "depth_before": before["depth"],
        "gates_1q_before": before["gates_1q"],
        "gates_2q_before": before["gates_2q"],
        "gate_types_before": before["gate_types"],
        "params_before": before["params"],
        "mid_measures_before": before["mid_measures"],
        "wires_before": before["wires"],

        "ops_after": after["ops"],
        "depth_after": after["depth"],
        "gates_1q_after": after["gates_1q"],
        "gates_2q_after": after["gates_2q"],
        "gate_types_after": after["gate_types"],
        "params_after": after["params"],
        "mid_measures_after": after["mid_measures"],
        "wires_after": after["wires"],

        "d_ops": after["ops"] - before["ops"],
        "d_depth": (after["depth"] - before["depth"]
                    if after["depth"] >= 0 and before["depth"] >= 0 else -1),
        "d_gates_1q": after["gates_1q"] - before["gates_1q"],
        "d_gates_2q": after["gates_2q"] - before["gates_2q"],
        "d_params": after["params"] - before["params"],
    })

    return new_tape


def run_pipeline(tape, pipeline, case_id, config="", program=""):
    """Apply a whole pipeline, logging every transform.

    `pipeline` is a list of (name, transform) or (name, transform, kwargs).
    """
    cur = tape
    for seq, entry in enumerate(pipeline):
        if len(entry) == 3:
            name, tf, kw = entry
        else:
            (name, tf), kw = entry, {}
        cur = apply_one(cur, name, tf, case_id, config, program, seq, **kw)
    return cur


def reset():
    """Clear recorded events."""
    EVENTS.clear()


def save(path):
    """Write all recorded events to CSV."""
    if not EVENTS:
        print("No events recorded.")
        return
    fields = list(EVENTS[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(EVENTS)
    print(f"\nWrote {len(EVENTS)} events x {len(fields)} columns to {path}")


def table(limit=20):
    """Print raw event rows."""
    if not EVENTS:
        print("No events recorded.")
        return
    hdr = (f"{'seq':>4} {'transform':<28}{'ms':>9}{'memKB':>8}{'chg':>5}"
           f"{'ops':>11}{'depth':>10}{'2q':>8}{'params':>9}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for e in EVENTS[:limit]:
        print(
            f"{e['seq']:>4} {e['activity'][:27]:<28}"
            f"{e['duration_ms']:>9.3f}{e['mem_alloc_kb']:>8.1f}"
            f"{str(e['changed'])[0]:>5}"
            f"{str(e['ops_before']) + '->' + str(e['ops_after']):>11}"
            f"{str(e['depth_before']) + '->' + str(e['depth_after']):>10}"
            f"{str(e['gates_2q_before']) + '->' + str(e['gates_2q_after']):>8}"
            f"{str(e['params_before']) + '->' + str(e['params_after']):>9}"
        )
    if len(EVENTS) > limit:
        print(f"  ... {len(EVENTS) - limit} more events")
    print()


def summary():
    """Per-transform totals: runs, no-ops, time, and net effect."""
    if not EVENTS:
        print("No events recorded.")
        return

    W = 84
    print("\n" + "=" * W)
    print("TRANSFORM EXECUTION SUMMARY")
    print("=" * W)

    names = sorted({e["activity"] for e in EVENTS})
    print(f"\n{'Transform':<30}{'runs':>6}{'no-op':>7}{'no-op%':>9}"
          f"{'ms':>10}{'memKB':>10}{'d_ops':>7}{'d_2q':>6}")
    print("-" * W)

    for n in names:
        rows = [e for e in EVENTS if e["activity"] == n]
        noop = sum(1 for e in rows if not e["changed"])
        ms = sum(e["duration_ms"] for e in rows)
        mem = sum(e["mem_alloc_kb"] for e in rows)
        d_ops = sum(e["d_ops"] for e in rows)
        d_2q = sum(e["d_gates_2q"] for e in rows)
        print(f"{n[:29]:<30}{len(rows):>6}{noop:>7}"
              f"{100 * noop / len(rows):>8.0f}%{ms:>10.2f}{mem:>10.1f}"
              f"{d_ops:>7}{d_2q:>6}")

    tot = len(EVENTS)
    noop_tot = sum(1 for e in EVENTS if not e["changed"])
    ms_tot = sum(e["duration_ms"] for e in EVENTS)
    mem_tot = sum(e["mem_alloc_kb"] for e in EVENTS)
    print("-" * W)
    print(f"{'TOTAL':<30}{tot:>6}{noop_tot:>7}"
          f"{100 * noop_tot / tot:>8.0f}%{ms_tot:>10.2f}{mem_tot:>10.1f}")

    waste = sum(e["duration_ms"] for e in EVENTS if not e["changed"])
    print(f"\nTime spent on transforms that changed nothing: "
          f"{waste:.2f} ms ({100 * waste / ms_tot:.0f}% of total)")

    failed = [e for e in EVENTS if e["failed"]]
    if failed:
        print(f"\nTransforms that raised: {len(failed)}")
        seen = {}
        for e in failed:
            seen[e["activity"]] = seen.get(e["activity"], 0) + 1
        for n, c in sorted(seen.items()):
            print(f"  {n}: {c}")

    by_time = sorted(names,
                     key=lambda n: -sum(e["duration_ms"] for e in EVENTS
                                        if e["activity"] == n))
    print("\nSlowest transforms:")
    for n in by_time[:5]:
        rows = [e for e in EVENTS if e["activity"] == n]
        ms = sum(e["duration_ms"] for e in rows)
        print(f"  {n[:29]:<30}{ms:>9.2f} ms  "
              f"({100 * ms / ms_tot:>4.1f}% of total, {len(rows)} runs)")
    print()
