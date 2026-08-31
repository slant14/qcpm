"""
pass_logger.py — records what Qiskit transpiler passes actually do.

Hooks into Qiskit's pass manager callback so every pass execution is
recorded as an event: which pass, when, how long it took, how much memory
it used, and how the circuit changed.

Import this and pass `make_callback()` to your pass manager.

    import pass_logger
    pm = generate_preset_pass_manager(optimization_level=2, ...)
    pm.run(circuit, callback=pass_logger.make_callback("my_case"))
    pass_logger.summary()
    pass_logger.save("pass_log.csv")
"""

import csv
import time
import tracemalloc
from datetime import datetime

# Every event recorded goes here.
EVENTS = []

# Set by make_callback so rows know what circuit they belong to.
_ctx = {}


def measure_dag(dag):
    """Measure a circuit DAG. Returns a dict of structural metrics."""
    blank = {
        "ops": -1, "depth": -1, "gates_2q": -1, "gates_1q": -1,
        "gate_types": -1, "qubits": -1, "clbits": -1,
    }
    try:
        nodes = dag.op_nodes()
    except Exception:
        return blank

    try:
        two_q = sum(1 for nd in nodes if len(nd.qargs) == 2)
        one_q = sum(1 for nd in nodes if len(nd.qargs) == 1)
        return {
            "ops": len(nodes),
            "depth": dag.depth(),
            "gates_2q": two_q,
            "gates_1q": one_q,
            "gate_types": len(dag.count_ops()),
            "qubits": dag.num_qubits(),
            "clbits": dag.num_clbits(),
        }
    except Exception:
        return blank


def make_callback(case_id, config="", program=""):
    """Build a callback for one compilation.

    IMPORTANT: Qiskit invokes the callback with KEYWORD arguments
    (pass_, dag, time, property_set, count). A positional signature
    raises TypeError. That is why the parameters below are all keyword
    with defaults.
    """
    _ctx.update(case_id=case_id, config=config, program=program,
                seq=0, prev=None)

    def callback(pass_=None, dag=None, time=None,
                 property_set=None, count=None):
        after = measure_dag(dag)
        # Qiskit hands us the DAG *after* the pass ran, so "before" is
        # whatever the previous pass left behind.
        before = _ctx["prev"] if _ctx["prev"] else after

        EVENTS.append({
            "case_id": _ctx["case_id"],
            "program": _ctx["program"],
            "config": _ctx["config"],
            "activity": pass_.name(),
            "timestamp": datetime.now().isoformat(),
            "seq": _ctx["seq"],

            "duration_ms": round((time or 0) * 1000, 4),

            "changed": before != after,

            "ops_before": before["ops"],
            "depth_before": before["depth"],
            "gates_1q_before": before["gates_1q"],
            "gates_2q_before": before["gates_2q"],
            "gate_types_before": before["gate_types"],
            "qubits_before": before["qubits"],
            "clbits_before": before["clbits"],

            "ops_after": after["ops"],
            "depth_after": after["depth"],
            "gates_1q_after": after["gates_1q"],
            "gates_2q_after": after["gates_2q"],
            "gate_types_after": after["gate_types"],
            "qubits_after": after["qubits"],
            "clbits_after": after["clbits"],

            "d_ops": after["ops"] - before["ops"],
            "d_depth": after["depth"] - before["depth"],
            "d_gates_1q": after["gates_1q"] - before["gates_1q"],
            "d_gates_2q": after["gates_2q"] - before["gates_2q"],
        })
        _ctx["prev"] = after
        _ctx["seq"] += 1

    return callback


def run_and_log(pm, circuit, case_id, config="", program=""):
    """Run a pass manager with logging, and record whole-run memory.

    Memory can't be split per pass from inside a callback, so the peak
    for the whole compilation is attached to the first row.
    """
    cb = make_callback(case_id, config, program)
    n0 = len(EVENTS)

    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        out = pm.run(circuit, callback=cb)
    finally:
        t1 = time.perf_counter()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    new = EVENTS[n0:]
    for e in new:
        e["mem_peak_kb"] = -1
    if new:
        new[0]["mem_peak_kb"] = round(peak / 1024, 2)

    return out, (t1 - t0) * 1000, len(new)


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
    hdr = (f"{'seq':>4} {'pass':<34}{'ms':>9}{'chg':>5}"
           f"{'ops':>12}{'depth':>10}{'2q':>9}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for e in EVENTS[:limit]:
        print(
            f"{e['seq']:>4} {e['activity'][:33]:<34}"
            f"{e['duration_ms']:>9.3f}{str(e['changed'])[0]:>5}"
            f"{str(e['ops_before']) + '->' + str(e['ops_after']):>12}"
            f"{str(e['depth_before']) + '->' + str(e['depth_after']):>10}"
            f"{str(e['gates_2q_before']) + '->' + str(e['gates_2q_after']):>9}"
        )
    if len(EVENTS) > limit:
        print(f"  ... {len(EVENTS) - limit} more events")
    print()


def summary():
    """Per-pass totals: runs, no-ops, time, and net effect."""
    if not EVENTS:
        print("No events recorded.")
        return

    W = 84
    print("\n" + "=" * W)
    print("PASS EXECUTION SUMMARY")
    print("=" * W)

    names = sorted({e["activity"] for e in EVENTS})
    print(f"\n{'Pass':<36}{'runs':>6}{'no-op':>7}{'no-op%':>9}"
          f"{'ms':>10}{'d_ops':>8}{'d_2q':>8}")
    print("-" * W)

    for n in names:
        rows = [e for e in EVENTS if e["activity"] == n]
        noop = sum(1 for e in rows if not e["changed"])
        ms = sum(e["duration_ms"] for e in rows)
        d_ops = sum(e["d_ops"] for e in rows)
        d_2q = sum(e["d_gates_2q"] for e in rows)
        print(f"{n[:35]:<36}{len(rows):>6}{noop:>7}"
              f"{100 * noop / len(rows):>8.0f}%{ms:>10.2f}{d_ops:>8}{d_2q:>8}")

    tot = len(EVENTS)
    noop_tot = sum(1 for e in EVENTS if not e["changed"])
    ms_tot = sum(e["duration_ms"] for e in EVENTS)
    print("-" * W)
    print(f"{'TOTAL':<36}{tot:>6}{noop_tot:>7}"
          f"{100 * noop_tot / tot:>8.0f}%{ms_tot:>10.2f}")

    waste = sum(e["duration_ms"] for e in EVENTS if not e["changed"])
    print(f"\nTime spent on passes that changed nothing: "
          f"{waste:.2f} ms ({100 * waste / ms_tot:.0f}% of total)")

    # Which passes eat the time
    by_time = sorted(names,
                     key=lambda n: -sum(e["duration_ms"] for e in EVENTS
                                        if e["activity"] == n))
    print("\nSlowest passes:")
    for n in by_time[:5]:
        rows = [e for e in EVENTS if e["activity"] == n]
        ms = sum(e["duration_ms"] for e in rows)
        print(f"  {n[:35]:<36}{ms:>9.2f} ms  "
              f"({100 * ms / ms_tot:>4.1f}% of total, {len(rows)} runs)")
    print()
