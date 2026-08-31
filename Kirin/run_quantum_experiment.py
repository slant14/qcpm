import pass_logger
from bloqade import qasm2
from bloqade.qasm2.passes import QASM2Fold, UOpToParallel

# ---------------------------------------------------------------
# 1. Quantum circuits to compile.
#    @qasm2.extended builds the circuit into Kirin IR.
# ---------------------------------------------------------------


@qasm2.extended
def ghz_state():
    """GHZ state: one Hadamard, then a chain of CNOTs. Nothing parallelisable."""
    q = qasm2.qreg(4)
    qasm2.h(q[0])
    qasm2.cx(q[0], q[1])
    qasm2.cx(q[1], q[2])
    qasm2.cx(q[2], q[3])
    return q


@qasm2.extended
def all_hadamards():
    """Six independent Hadamards. On neutral atoms these could run at once."""
    q = qasm2.qreg(6)
    qasm2.h(q[0])
    qasm2.h(q[1])
    qasm2.h(q[2])
    qasm2.h(q[3])
    qasm2.h(q[4])
    qasm2.h(q[5])
    return q


@qasm2.extended
def rotations():
    """Rotation gates with angles computed at compile time."""
    q = qasm2.qreg(4)
    theta = 3.14159 / 2
    qasm2.rz(q[0], theta)
    qasm2.rz(q[1], theta * 2)
    qasm2.rz(q[2], theta / 2)
    qasm2.rz(q[3], theta + 1.0)
    return q


@qasm2.extended
def loop_circuit():
    """A circuit built in a loop — the compiler has to unroll it."""
    q = qasm2.qreg(5)
    for i in range(5):
        qasm2.h(q[i])
    return q


@qasm2.extended
def already_minimal():
    """Two gates. Almost nothing for the compiler to do."""
    q = qasm2.qreg(2)
    qasm2.h(q[0])
    qasm2.cx(q[0], q[1])
    return q


CIRCUITS = [
    ghz_state,
    all_hadamards,
    rotations,
    loop_circuit,
    already_minimal,
]


# ---------------------------------------------------------------
# 2. Compile each circuit with logging on.
# ---------------------------------------------------------------

def gate_count(method):
    """Count actual quantum gate operations in the IR."""
    n = 0
    try:
        for stmt in method.callable_region.walk():
            name = stmt.name
            dialect = getattr(stmt, "dialect", None)
            dialect_name = getattr(dialect, "name", "")
            if "uop" in dialect_name or "parallel" in dialect_name:
                n += 1
    except Exception:
        return -1
    return n


def main():
    pass_logger.enable()

    for circuit in CIRCUITS:
        print(f"\n{'=' * 62}")
        print(f"COMPILING CIRCUIT: {circuit.sym_name}")
        print("=" * 62)

        print(f"\ngates before: {gate_count(circuit)}")
        print("\n--- IR before passes ---")
        circuit.print()

        # A small neutral-atom compilation pipeline.
        QASM2Fold(circuit.dialects)(circuit)
        UOpToParallel(circuit.dialects)(circuit)

        print(f"\ngates after: {gate_count(circuit)}")
        print("\n--- IR after passes ---")
        circuit.print()

    pass_logger.disable()

    pass_logger.table()
    pass_logger.save("quantum_pass_log.csv")


if __name__ == "__main__":
    main()
