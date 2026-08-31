import csv
import time
import tracemalloc
from datetime import datetime

from kirin.passes.abc import Pass

EVENTS = []

_original_call = Pass.__call__
_original_fixpoint = Pass.fixpoint

# Passes call other passes. tracemalloc can only be started once, so we
# track depth and let only the outermost pass control tracing.
_depth = 0


# ---------------------------------------------------------------
# IR measurement
# ---------------------------------------------------------------

def measure_ir(mt):
    """Measure a method's IR. Returns -1 for every field on failure."""
    blank = {
        "stmts": -1, "blocks": -1, "ssa_values": -1, "edges": -1,
        "dialects": -1, "dead_values": -1, "max_depth": -1,
        "constants": -1, "quantum_gates": -1,
    }

    try:
        region = mt.callable_region
        stmts = list(region.walk())
    except Exception:
        return blank

    try:
        def depth_of(node, d=0):
            best = d
            try:
                for r in node.regions:
                    for b in r.blocks:
                        for s in b.stmts:
                            best = max(best, depth_of(s, d + 1))
            except Exception:
                pass
            return best

        max_depth = 0
        for b in region.blocks:
            for s in b.stmts:
                max_depth = max(max_depth, depth_of(s, 1))

        return {
            "stmts": len(stmts),
            "blocks": len(region.blocks),
            "ssa_values": sum(len(s.results) for s in stmts),
            "edges": sum(len(s.args) for s in stmts),
            "dialects": len({getattr(s.dialect, "name", "?") for s in stmts}),
            "dead_values": sum(
                1 for s in stmts for v in s.results if len(v.uses) == 0
            ),
            "max_depth": max_depth,
            "constants": sum(
                1 for s in stmts if "constant" in s.name.lower()
            ),
            "quantum_gates": sum(
                1 for s in stmts
                if any(k in getattr(s.dialect, "name", "")
                       for k in ("uop", "parallel", "glob"))
            ),
        }
    except Exception:
        return blank


# ---------------------------------------------------------------
# Recording
# ---------------------------------------------------------------

def _record(pass_obj, mt, before, after, elapsed_ms,
            mem_delta, mem_peak, result, iteration, nesting):
    """Append one row of measurements."""
    EVENTS.append({
        "case_id": mt.sym_name,
        "activity": type(pass_obj).__name__,
        "timestamp": datetime.fromtimestamp(time.time()).isoformat(),
        "iteration": iteration,
        "nesting_level": nesting,

        "duration_ms": round(elapsed_ms, 4),
        "mem_alloc_kb": round(mem_delta / 1024, 2),
        "mem_peak_kb": round(mem_peak / 1024, 2),

        "changed": bool(getattr(result, "has_done_something", False)),

        "stmts_before": before["stmts"],
        "blocks_before": before["blocks"],
        "ssa_before": before["ssa_values"],
        "edges_before": before["edges"],
        "dialects_before": before["dialects"],
        "dead_before": before["dead_values"],
        "depth_before": before["max_depth"],
        "consts_before": before["constants"],
        "gates_before": before["quantum_gates"],

        "stmts_after": after["stmts"],
        "blocks_after": after["blocks"],
        "ssa_after": after["ssa_values"],
        "edges_after": after["edges"],
        "dialects_after": after["dialects"],
        "dead_after": after["dead_values"],
        "depth_after": after["max_depth"],
        "consts_after": after["constants"],
        "gates_after": after["quantum_gates"],

        "d_stmts": after["stmts"] - before["stmts"],
        "d_blocks": after["blocks"] - before["blocks"],
        "d_ssa": after["ssa_values"] - before["ssa_values"],
        "d_edges": after["edges"] - before["edges"],
        "d_dead": after["dead_values"] - before["dead_values"],
        "d_consts": after["constants"] - before["constants"],
        "d_gates": after["quantum_gates"] - before["quantum_gates"],
    })


def _instrumented(run_fn, pass_obj, mt, iteration):
    """Run one pass with measurement around it."""
    global _depth

    before = measure_ir(mt)

    outermost = _depth == 0
    if outermost:
        tracemalloc.start()

    snap_before = tracemalloc.take_snapshot()
    _depth += 1
    t0 = time.perf_counter()

    try:
        result = run_fn()
    finally:
        t1 = time.perf_counter()
        _depth -= 1

    try:
        snap_after = tracemalloc.take_snapshot()
        _, peak = tracemalloc.get_traced_memory()
        mem_delta = sum(
            s.size_diff for s in snap_after.compare_to(snap_before, "filename")
        )
    except Exception:
        mem_delta, peak = 0, 0

    if outermost:
        tracemalloc.stop()

    after = measure_ir(mt)
    _record(pass_obj, mt, before, after, (t1 - t0) * 1000,
            mem_delta, peak, result, iteration, _depth)
    return result


def _logged_call(self, mt):
    return _instrumented(lambda: _original_call(self, mt), self, mt, 1)


def _logged_fixpoint(self, mt, max_iter=32):
    """Kirin's fixpoint loop, with every iteration recorded separately."""
    from kirin.rewrite.abc import RewriteResult

    result = RewriteResult()
    for i in range(max_iter):
        result_ = _instrumented(lambda: self.unsafe_run(mt), self, mt, i + 1)
        result = result_.join(result)
        if not result_.has_done_something:
            break
    mt.verify()
    return result


# ---------------------------------------------------------------
# Control
# ---------------------------------------------------------------

def enable():
    """Turn logging on."""
    Pass.__call__ = _logged_call
    Pass.fixpoint = _logged_fixpoint


def disable():
    """Turn logging off."""
    Pass.__call__ = _original_call
    Pass.fixpoint = _original_fixpoint


def reset():
    """Clear recorded events."""
    EVENTS.clear()


def save(path="pass_log.csv"):
    """Write all recorded events to CSV."""
    if not EVENTS:
        print("No events recorded. Did you call enable() first?")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(EVENTS[0].keys()))
        w.writeheader()
        w.writerows(EVENTS)
    print(f"Wrote {len(EVENTS)} rows x {len(EVENTS[0])} columns to {path}")


def table():
    """Print the raw rows. No aggregation."""
    if not EVENTS:
        print("No events recorded.")
        return

    cols = ["case_id", "activity", "iteration", "nesting_level",
            "duration_ms", "mem_alloc_kb", "mem_peak_kb", "changed",
            "stmts_before", "stmts_after", "d_stmts",
            "ssa_before", "ssa_after", "blocks_before", "blocks_after",
            "dead_before", "dead_after", "consts_before", "consts_after",
            "gates_before", "gates_after"]

    hdr = (f"{'case':<18}{'pass':<20}{'it':>3}{'lvl':>4}"
           f"{'ms':>9}{'memKB':>9}{'peakKB':>9}{'chg':>5}"
           f"{'stmt':>10}{'ssa':>9}{'blk':>7}{'dead':>7}"
           f"{'const':>8}{'gate':>8}")
    print("\n" + hdr)
    print("-" * len(hdr))

    for e in EVENTS:
        print(
            f"{e['case_id'][:17]:<18}{e['activity'][:19]:<20}"
            f"{e['iteration']:>3}{e['nesting_level']:>4}"
            f"{e['duration_ms']:>9.2f}{e['mem_alloc_kb']:>9.1f}"
            f"{e['mem_peak_kb']:>9.1f}{str(e['changed'])[0]:>5}"
            f"{str(e['stmts_before']) + '->' + str(e['stmts_after']):>10}"
            f"{str(e['ssa_before']) + '->' + str(e['ssa_after']):>9}"
            f"{str(e['blocks_before']) + '->' + str(e['blocks_after']):>7}"
            f"{str(e['dead_before']) + '->' + str(e['dead_after']):>7}"
            f"{str(e['consts_before']) + '->' + str(e['consts_after']):>8}"
            f"{str(e['gates_before']) + '->' + str(e['gates_after']):>8}"
        )
    print()
