import math

import pass_logger
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

N = 4
BASIS = ["u", "cx"]


# ---------------------------------------------------------------
# 1. Pure quantum circuits.
# ---------------------------------------------------------------

def ghz():
    """Maximally entangled state: one H, then a chain of CNOTs."""
    qc = QuantumCircuit(N)
    qc.h(0)
    for i in range(N - 1):
        qc.cx(i, i + 1)
    return qc


def qft():
    """Quantum Fourier transform. Rotation-heavy, densely connected."""
    qc = QuantumCircuit(N)
    for j in range(N):
        qc.h(j)
        for k in range(j + 1, N):
            qc.cp(math.pi / (2 ** (k - j)), j, k)
    return qc


def full_entangle():
    """Every qubit entangled with every other. Worst case for routing."""
    qc = QuantumCircuit(N)
    for i in range(N):
        qc.h(i)
    for i in range(N):
        for j in range(i + 1, N):
            qc.cz(i, j)
    return qc


def redundant():
    """Every gate has its inverse immediately after.

    This is the CONTROL circuit. It provably reduces to nothing, so any
    working optimizer must collapse it. If a pipeline doesn't, that tells
    you something about the pipeline.
    """
    qc = QuantumCircuit(N)
    for i in range(N):
        qc.h(i)
        qc.h(i)
        qc.x(i)
        qc.x(i)
    for i in range(N - 1):
        qc.cx(i, i + 1)
        qc.cx(i, i + 1)
    return qc


def rotations():
    """Stacked single-qubit rotations. Should merge into fewer gates."""
    qc = QuantumCircuit(N)
    for i in range(N):
        for k in range(4):
            qc.rz(0.1 * (k + 1), i)
        for k in range(3):
            qc.ry(0.2 * (k + 1), i)
    return qc


def layered():
    """VQE-style ansatz: rotation layer, entangling layer, repeated.

    The realistic workload — this is the shape of circuit that gets
    recompiled thousands of times in a variational algorithm.
    """
    qc = QuantumCircuit(N)
    for layer in range(3):
        for i in range(N):
            qc.ry(0.1 * (layer + 1) * (i + 1), i)
            qc.rz(0.2 * (layer + 1) * (i + 1), i)
        for i in range(N - 1):
            qc.cx(i, i + 1)
    return qc


CIRCUITS = [
    ("ghz", ghz),
    ("qft", qft),
    ("full_entangle", full_entangle),
    ("redundant", redundant),
    ("rotations", rotations),
    ("layered", layered),
]


# ---------------------------------------------------------------
# 2. Hardware topologies.
# ---------------------------------------------------------------

ARCHITECTURES = {
    # No connectivity constraint. The control condition — routing passes
    # have nothing to do.
    "none": None,
    # A straight line: 0-1-2-3. Worst case, maximum SWAP insertion.
    "linear": [[i, i + 1] for i in range(N - 1)],
    # A ring: linear plus a wraparound edge.
    "ring": [[i, (i + 1) % N] for i in range(N)],
}


# ---------------------------------------------------------------
# 3. Compile everything.
# ---------------------------------------------------------------

def main():
    print("\n" + "=" * 70)
    print("QISKIT — QUANTUM CIRCUITS")
    print("=" * 70)

    for name, builder in CIRCUITS:
        qc = builder()
        two_q = sum(1 for inst in qc.data if len(inst.qubits) == 2)
        print(f"\n  {name}")
        print(f"    built: {len(qc.data)} gates, depth {qc.depth()}, "
              f"{two_q} two-qubit gates")

        for level in [0, 1, 2, 3]:
            for arch_name, coupling in ARCHITECTURES.items():
                pm = generate_preset_pass_manager(
                    optimization_level=level,
                    basis_gates=BASIS,
                    coupling_map=coupling,
                )
                try:
                    _, ms, n_passes = pass_logger.run_and_log(
                        pm, qc.copy(),
                        case_id=f"{name}_L{level}_{arch_name}",
                        config=f"level{level}_{arch_name}",
                        program=name,
                    )
                    print(f"    level {level} / {arch_name:<7} "
                          f"{n_passes:>3} passes, {ms:>7.2f} ms")
                except Exception as e:
                    print(f"    level {level} / {arch_name:<7} FAILED "
                          f"{type(e).__name__}: {str(e)[:40]}")

    pass_logger.summary()
    pass_logger.table()
    pass_logger.save("quantum_pass_log.csv")


if __name__ == "__main__":
    main()
