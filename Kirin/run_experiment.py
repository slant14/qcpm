import pass_logger
from kirin.prelude import basic_no_opt
from kirin.passes import Fold, TypeInfer
from kirin.passes.canonicalize import Canonicalize

# ---------------------------------------------------------------
# 1. Sample programs to compile.
#    basic_no_opt means "build the IR but don't optimize it yet",
#    so we can watch the passes do the work ourselves.
# ---------------------------------------------------------------


@basic_no_opt
def lots_of_constants(x: int) -> int:
    """Should fold heavily: most of this is computable at compile time."""
    a = 2 + 3
    b = a * 4
    c = b - 7
    d = c + 100
    return d + x


@basic_no_opt
def dead_code(x: int) -> int:
    """Has unused variables that dead-code elimination should remove."""
    unused_1 = 111
    unused_2 = 222
    unused_3 = unused_1 + unused_2
    return x + 1


@basic_no_opt
def nothing_to_do(x: int, y: int) -> int:
    """Already minimal. Passes should mostly no-op here."""
    return x + y


@basic_no_opt
def with_a_loop(n: int) -> int:
    """Has control flow, so more passes get involved."""
    total = 0
    for i in range(n):
        total = total + i * 2
    return total


PROGRAMS = [lots_of_constants, dead_code, nothing_to_do, with_a_loop]


# ---------------------------------------------------------------
# 2. Run passes over each program, with logging on.
# ---------------------------------------------------------------

def main():
    pass_logger.enable()

    for method in PROGRAMS:
        print(f"\n{'=' * 62}")
        print(f"COMPILING: {method.sym_name}")
        print("=" * 62)

        print("\n--- IR before any passes ---")
        method.print()

        # Run three passes in sequence. This is our mini pipeline.
        Canonicalize(method.dialects).fixpoint(method)
        TypeInfer(method.dialects)(method)
        Fold(method.dialects).fixpoint(method)

        print("\n--- IR after passes ---")
        method.print()

    pass_logger.disable()

    pass_logger.table()
    pass_logger.save("pass_log.csv")


if __name__ == "__main__":
    main()
