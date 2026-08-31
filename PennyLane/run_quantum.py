"""
run_quantum.py — apply transforms to pure quantum circuits and log each one.

No mid-circuit measurement, no conditionals, no classical arithmetic in
the parameters. Just gates.

This adds an axis the classical run doesn't have: DECOMPOSITION TO A GATE
SET. Real hardware supports a limited set of gates, so anything else has
to be rewritten in terms of what's available. Comparing a pipeline with
and without decomposition shows how much that costs.

Run with:   python run_quantum.py
"""

import math

import pennylane as qml
import pass_logger

N = 4


# ---------------------------------------------------------------
# 1. Pure quantum circuits.
# ---------------------------------------------------------------

def build(ops):
    """Wrap a list of gates into a tape with a standard measurement."""
    return qml.tape.QuantumScript(ops, [qml.expval(qml.PauliZ(0))])


def ghz():
    """Maximally entangled state: one H, then a chain of CNOTs."""
    ops = [qml.Hadamard(0)]
    for i in range(N - 1):
        ops.append(qml.CNOT([i, i + 1]))
    return build(ops)


def qft():
    """Quantum Fourier transform. Rotation-heavy, densely connected."""
    ops = []
    for j in range(N):
        ops.append(qml.Hadamard(j))
        for k in range(j + 1, N):
            ops.append(qml.ControlledPhaseShift(
                math.pi / (2 ** (k - j)), [j, k]))
    return build(ops)


def full_entangle():
    """Every qubit entangled with every other."""
    ops = [qml.Hadamard(i) for i in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            ops.append(qml.CZ([i, j]))
    return build(ops)


def redundant():
    """Every gate has its inverse immediately after.

    The CONTROL circuit. It provably reduces to nothing, so any working
    optimizer must collapse it.
    """
    ops = []
    for i in range(N):
        ops += [qml.Hadamard(i), qml.Hadamard(i),
                qml.PauliX(i), qml.PauliX(i)]
    for i in range(N - 1):
        ops += [qml.CNOT([i, i + 1]), qml.CNOT([i, i + 1])]
    return build(ops)


def rotations():
    """Stacked single-qubit rotations. Should merge into fewer gates."""
    ops = []
    for i in range(N):
        for k in range(4):
            ops.append(qml.RZ(0.1 * (k + 1), i))
        for k in range(3):
            ops.append(qml.RY(0.2 * (k + 1), i))
    return build(ops)


def layered():
    """VQE-style ansatz: rotation layer, entangling layer, repeated.

    The realistic workload — this shape gets recompiled thousands of
    times inside a variational optimization loop.
    """
    ops = []
    for layer in range(3):
        for i in range(N):
            ops.append(qml.RY(0.1 * (layer + 1) * (i + 1), i))
            ops.append(qml.RZ(0.2 * (layer + 1) * (i + 1), i))
        for i in range(N - 1):
            ops.append(qml.CNOT([i, i + 1]))
    return build(ops)


def with_swaps():
    """Contains SWAP gates, which undo_swaps should be able to remove."""
    ops = [qml.Hadamard(0)]
    for i in range(N - 1):
        ops.append(qml.CNOT([i, i + 1]))
        ops.append(qml.SWAP([i, i + 1]))
    return build(ops)


CIRCUITS = [
    ("ghz", ghz),
    ("qft", qft),
    ("full_entangle", full_entangle),
    ("redundant", redundant),
    ("rotations", rotations),
    ("layered", layered),
    ("with_swaps", with_swaps),
]


# ---------------------------------------------------------------
# 2. Transform pipelines.
#
# NOTE: these are OURS. PennyLane ships no preset optimization levels.
# ---------------------------------------------------------------

# A hardware-plausible target gate set.
TARGET_GATES = {qml.RZ, qml.RY, qml.RX, qml.CNOT}

PIPELINES = {
    "light": [
        ("cancel_inverses", qml.transforms.cancel_inverses),
        ("merge_rotations", qml.transforms.merge_rotations),
    ],
    "medium": [
        ("cancel_inverses", qml.transforms.cancel_inverses),
        ("merge_rotations", qml.transforms.merge_rotations),
        ("single_qubit_fusion", qml.transforms.single_qubit_fusion),
        ("commute_controlled", qml.transforms.commute_controlled),
        ("undo_swaps", qml.transforms.undo_swaps),
    ],
    "heavy": [
        ("cancel_inverses", qml.transforms.cancel_inverses),
        ("merge_rotations", qml.transforms.merge_rotations),
        ("single_qubit_fusion", qml.transforms.single_qubit_fusion),
        ("commute_controlled", qml.transforms.commute_controlled),
        ("undo_swaps", qml.transforms.undo_swaps),
        ("cancel_inverses_2", qml.transforms.cancel_inverses),
        ("merge_rotations_2", qml.transforms.merge_rotations),
    ],
    # Decomposition to a fixed gate set — the hardware-targeting axis.
    "decompose": [
        ("cancel_inverses", qml.transforms.cancel_inverses),
        ("decompose", qml.transforms.decompose,
         {"gate_set": TARGET_GATES}),
        ("merge_rotations", qml.transforms.merge_rotations),
        ("single_qubit_fusion", qml.transforms.single_qubit_fusion),
    ],
}


def main():
    print("\n" + "=" * 70)
    print("PENNYLANE — QUANTUM CIRCUITS")
    print("=" * 70)

    for name, builder in CIRCUITS:
        tape = builder()
        two_q = sum(1 for o in tape.operations if len(o.wires) == 2)
        print(f"\n  {name}")
        print(f"    built: {len(tape.operations)} gates, "
              f"{two_q} two-qubit gates, {len(tape.wires)} wires")

        for pipe_name, pipeline in PIPELINES.items():
            fresh = builder()
            before = len(fresh.operations)
            out = pass_logger.run_pipeline(
                fresh, pipeline,
                case_id=f"{name}_{pipe_name}",
                config=pipe_name,
                program=name,
            )
            print(f"    {pipe_name:<10} {len(pipeline)} transforms, "
                  f"ops {before} -> {len(out.operations)}")

    pass_logger.summary()
    pass_logger.table()
    pass_logger.save("quantum_pass_log.csv")


if __name__ == "__main__":
    main()
