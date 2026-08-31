import math

import pass_logger
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

N = 4
BASIS = ["u", "cx"]


# ---------------------------------------------------------------
# 1. Programs with classical structure.
# ---------------------------------------------------------------

def measure_only():
    """Classical registers and measurement, no conditionals."""
    q, c = QuantumRegister(N, "q"), ClassicalRegister(N, "c")
    qc = QuantumCircuit(q, c)
    qc.h(0)
    for i in range(N - 1):
        qc.cx(i, i + 1)
    qc.measure(q, c)
    return qc


def conditional():
    """A gate that only runs if a measured bit came out 1."""
    q, c = QuantumRegister(N, "q"), ClassicalRegister(N, "c")
    qc = QuantumCircuit(q, c)
    qc.h(0)
    qc.measure(0, 0)
    with qc.if_test((c[0], 1)):
        qc.x(1)
    qc.measure(q, c)
    return qc


def feedforward():
    """Measure, condition on it, measure again, condition again."""
    q, c = QuantumRegister(N, "q"), ClassicalRegister(N, "c")
    qc = QuantumCircuit(q, c)
    for i in range(N):
        qc.h(i)
    qc.measure(0, 0)
    with qc.if_test((c[0], 1)):
        qc.x(1)
    qc.cx(1, 2)
    qc.measure(2, 2)
    with qc.if_test((c[2], 1)):
        qc.z(3)
    qc.measure(q, c)
    return qc


def param_arithmetic():
    """Gate angles computed by classical arithmetic at build time."""
    q, c = QuantumRegister(N, "q"), ClassicalRegister(N, "c")
    qc = QuantumCircuit(q, c)
    base = math.pi / 4
    for i in range(N):
        theta = base * (i + 1) + 0.1 * i
        qc.rz(theta, i)
        qc.ry(theta / 2, i)
    for i in range(N - 1):
        qc.cx(i, i + 1)
    qc.measure(q, c)
    return qc


def reset_reuse():
    """Mid-circuit reset, then the qubit is used again."""
    q, c = QuantumRegister(N, "q"), ClassicalRegister(N, "c")
    qc = QuantumCircuit(q, c)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure(0, 0)
    qc.reset(0)
    qc.h(0)
    qc.cx(0, 2)
    qc.measure(q, c)
    return qc


def no_classical():
    """Control: no registers, no measurement, no conditionals.

    Comparing this against the others shows which passes only fire when
    classical structure is present.
    """
    qc = QuantumCircuit(N)
    qc.h(0)
    for i in range(N - 1):
        qc.cx(i, i + 1)
    return qc


PROGRAMS = [
    ("measure_only", measure_only),
    ("conditional", conditional),
    ("feedforward", feedforward),
    ("param_arithmetic", param_arithmetic),
    ("reset_reuse", reset_reuse),
    ("no_classical", no_classical),
]


# ---------------------------------------------------------------
# 2. Compile each program at every optimization level.
# ---------------------------------------------------------------

def main():
    print("\n" + "=" * 70)
    print("QISKIT — CLASSICAL STRUCTURE")
    print("=" * 70)

    for name, builder in PROGRAMS:
        print(f"\n  {name}")
        qc = builder()
        print(f"    built: {len(qc.data)} instructions, "
              f"depth {qc.depth()}, {qc.num_clbits} classical bits")

        for level in [0, 1, 2, 3]:
            pm = generate_preset_pass_manager(
                optimization_level=level, basis_gates=BASIS
            )
            try:
                _, ms, n_passes = pass_logger.run_and_log(
                    pm, qc.copy(),
                    case_id=f"{name}_L{level}",
                    config=f"level{level}",
                    program=name,
                )
                print(f"    level {level}: {n_passes:>3} passes, {ms:>7.2f} ms")
            except Exception as e:
                print(f"    level {level}: FAILED "
                      f"{type(e).__name__}: {str(e)[:45]}")

    pass_logger.summary()
    pass_logger.table()
    pass_logger.save("classical_pass_log.csv")


if __name__ == "__main__":
    main()
