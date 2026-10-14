
import time
from scipy.optimize import minimize
from beam_model import (
    evaluate_beam,
    WIDTH_BOUNDS_MM,
    HEIGHT_BOUNDS_MM,
    MAX_DEFLECTION_MM,
    MAX_VON_MISES_STRESS_MPA,
)

# Count how many times the provided evaluator is called
evaluation_count = 0


def evaluate_design(x):
    """Evaluate one candidate using the professor's beam model."""
    global evaluation_count
    evaluation_count += 1

    width_mm, height_mm = x
    return evaluate_beam(float(width_mm), float(height_mm))


def objective(x):
    """Minimize beam mass."""
    return evaluate_design(x)["mass_kg"]


def deflection_constraint(x):
    result = evaluate_design(x)
    return MAX_DEFLECTION_MM - result["max_deflection_mm"] - 1e-4


def stress_constraint(x):
    result = evaluate_design(x)
    return MAX_VON_MISES_STRESS_MPA - result["max_von_mises_stress_MPa"] - 1e-3


def main():
    global evaluation_count

    start_time = time.perf_counter()

    result = minimize(
        fun=objective,
        x0=[50.0, 80.0],
        method="SLSQP",
        bounds=[WIDTH_BOUNDS_MM, HEIGHT_BOUNDS_MM],
        constraints=[
            {"type": "ineq", "fun": deflection_constraint},
            {"type": "ineq", "fun": stress_constraint},
        ],
        options={"maxiter": 500, "ftol": 1e-12},
    )

    elapsed_time = time.perf_counter() - start_time

    width_mm, height_mm = result.x
    final_design = evaluate_beam(float(width_mm), float(height_mm))

    print("\nNUMERICAL OPTIMIZER RESULTS")
    print("---------------------------")
    print(f"Width: {width_mm:.4f} mm")
    print(f"Height: {height_mm:.4f} mm")
    print(f"Mass: {final_design['mass_kg']:.4f} kg")
    print(
        f"Deflection: "
        f"{final_design['max_deflection_mm']:.4f} mm"
    )
    print(
        f"Stress: "
        f"{final_design['max_von_mises_stress_MPa']:.4f} MPa"
    )
    print(f"Feasible: {final_design['feasible']}")
    print(f"Optimizer success: {result.success}")
    print(f"Optimizer message: {result.message}")
    print(f"Function evaluations reported: {result.nfev}")
    print(f"Evaluator calls counted: {evaluation_count}")
    print(f"Elapsed time: {elapsed_time:.6f} seconds")


if __name__ == "__main__":
    main()
