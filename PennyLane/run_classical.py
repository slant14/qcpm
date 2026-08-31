"""
run_classical.py — compile tapes containing CLASSICAL structure and log
every transform.

What "classical" means here: gate angles computed by classical arithmetic
before the tape is built, mid-circuit measurement (qml.measure), and
classical conditioning (qml.cond).

PennyLane is a hybrid framework, so this is closer to its natural mode
than it is for a pure quantum compiler. But note it is still not
classical *program* compilation — PennyLane does not compile ordinary
Python the way a general compiler infrastructure does.

Run with:   python run_classical.py
"""

import math

import pennylane as qml
import pass_logger

N = 4


# ---------------------------------------------------------------
# 1. Programs with classical structure.
#
# AnnotatedQueue is used rather than building an op list directly,
# because qml.measure and qml.cond only record properly inside a queue.
# ---------------------------------------------------------------

def param_arithmetic():
    """Angles computed by classical arithmetic at build time."""
    with qml.queuing.AnnotatedQueue() as q:
        base = math.pi / 4
        for i in range(N):
            theta = base * (i + 1) + 0.1 * i
            qml.RZ(theta, i)
            qml.RY(theta / 2, i)
        for i in range(N - 1):
            qml.CNOT([i, i + 1])
        qml.expval(qml.PauliZ(0))
    return qml.tape.QuantumScript.from_queue(q)


def deep_arithmetic():
    """Angles from a nested classical computation."""
    with qml.queuing.AnnotatedQueue() as q:
        for i in range(N):
            a = (i + 1) * 0.1
            b = a * 2 + 0.05
            c = math.sin(b) + math.cos(a)
            d = c / 2 + b * 3
            qml.RX(d, i)
            qml.RY(d / 3, i)
            qml.RZ(d * 0.5, i)
        qml.expval(qml.PauliZ(0))
    return qml.tape.QuantumScript.from_queue(q)


def mid_measure():
    """Mid-circuit measurement, no conditioning on it."""
    with qml.queuing.AnnotatedQueue() as q:
        qml.Hadamard(0)
        qml.CNOT([0, 1])
        qml.measure(0)
        qml.Hadamard(2)
        qml.CNOT([2, 3])
        qml.expval(qml.PauliZ(1))
    return qml.tape.QuantumScript.from_queue(q)


def conditional():
    """A gate that only runs if the measurement came out 1."""
    with qml.queuing.AnnotatedQueue() as q:
        qml.Hadamard(0)
        m = qml.measure(0)
        qml.cond(m, qml.PauliX)(1)
        qml.CNOT([1, 2])
        qml.expval(qml.PauliZ(2))
    return qml.tape.QuantumScript.from_queue(q)


def feedforward():
    """Measure, condition, measure again, condition again."""
    with qml.queuing.AnnotatedQueue() as q:
        for i in range(N):
            qml.Hadamard(i)
        m0 = qml.measure(0)
        qml.cond(m0, qml.PauliX)(1)
        qml.CNOT([1, 2])
        m1 = qml.measure(2)
        qml.cond(m1, qml.PauliZ)(3)
        qml.expval(qml.PauliZ(3))
    return qml.tape.QuantumScript.from_queue(q)


def cancelling_params():
    """Angles computed classically so that adjacent rotations cancel.

    This is a test of whether the transforms reason about VALUES or only
    about structure. The angles sum to zero, so merge_rotations should
    collapse each pair.
    """
    with qml.queuing.AnnotatedQueue() as q:
        for i in range(N):
            theta = 0.3 * (i + 1)
            qml.RZ(theta, i)
            qml.RZ(-theta, i)
            qml.RY(theta, i)
            qml.RY(-theta, i)
        qml.expval(qml.PauliZ(0))
    return qml.tape.QuantumScript.from_queue(q)


def no_classical():
    """Control: plain gates, no classical structure at all."""
    with qml.queuing.AnnotatedQueue() as q:
        qml.Hadamard(0)
        for i in range(N - 1):
            qml.CNOT([i, i + 1])
        qml.expval(qml.PauliZ(0))
    return qml.tape.QuantumScript.from_queue(q)


PROGRAMS = [
    ("param_arithmetic", param_arithmetic),
    ("deep_arithmetic", deep_arithmetic),
    ("mid_measure", mid_measure),
    ("conditional", conditional),
    ("feedforward", feedforward),
    ("cancelling_params", cancelling_params),
    ("no_classical", no_classical),
]


# ---------------------------------------------------------------
# 2. Transform pipelines.
#
# NOTE: these are OURS. PennyLane ships no preset optimization levels.
# ---------------------------------------------------------------

PIPELINES = {
    "light": [
        ("cancel_inverses", qml.transforms.cancel_inverses),
        ("merge_rotations", qml.transforms.merge_rotations),
    ],
    "medium": [
        ("cancel_inverses", qml.transforms.cancel_inverses),
        ("merge_rotations", qml.transforms.merge_rotations),
        ("single_qubit_fusion", qml.transforms.single_qubit_fusion),
    ],
    "heavy": [
        ("cancel_inverses", qml.transforms.cancel_inverses),
        ("merge_rotations", qml.transforms.merge_rotations),
        ("single_qubit_fusion", qml.transforms.single_qubit_fusion),
        ("commute_controlled", qml.transforms.commute_controlled),
        ("undo_swaps", qml.transforms.undo_swaps),
        # Deliberately repeated: does a second pass find anything the
        # first missed, once other transforms have run?
        ("cancel_inverses_2", qml.transforms.cancel_inverses),
    ],
}


def main():
    print("\n" + "=" * 70)
    print("PENNYLANE — CLASSICAL STRUCTURE")
    print("=" * 70)

    for name, builder in PROGRAMS:
        tape = builder()
        n_mid = sum(1 for o in tape.operations
                    if type(o).__name__ in ("MidMeasure", "MidMeasureMP",
                                            "Conditional"))
        print(f"\n  {name}")
        print(f"    built: {len(tape.operations)} ops, "
              f"{len(tape.wires)} wires, {n_mid} mid-circuit ops")

        for pipe_name, pipeline in PIPELINES.items():
            fresh = builder()
            before = len(fresh.operations)
            out = pass_logger.run_pipeline(
                fresh, pipeline,
                case_id=f"{name}_{pipe_name}",
                config=pipe_name,
                program=name,
            )
            print(f"    {pipe_name:<8} {len(pipeline)} transforms, "
                  f"ops {before} -> {len(out.operations)}")

    pass_logger.summary()
    pass_logger.table()
    pass_logger.save("classical_pass_log.csv")


if __name__ == "__main__":
    main()
