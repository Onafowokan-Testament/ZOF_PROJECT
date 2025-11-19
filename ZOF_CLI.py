from abc import ABC, abstractmethod


class RootFinder(ABC):
    """Abstract base class for root-finding methods"""

    def __init__(self, tolerance=1e-6, max_iterations=100):
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.iterations = []

    @abstractmethod
    def solve(self, **kwargs):
        pass

    def f(self, x, coefficients):
        """Evaluate polynomial: a0 + a1*x + a2*x^2 + ... + an*x^n"""
        result = 0
        for i, coeff in enumerate(coefficients):
            result += coeff * (x**i)
        return result

    def df(self, x, coefficients):
        """Derivative of polynomial"""
        result = 0
        for i in range(1, len(coefficients)):
            result += i * coefficients[i] * (x ** (i - 1))
        return result


class BisectionMethod(RootFinder):
    """Bisection Method for root finding"""

    def solve(self, a, b, coefficients):
        self.iterations = []

        if self.f(a, coefficients) * self.f(b, coefficients) > 0:
            return None, "Error: f(a) and f(b) must have opposite signs"

        iteration = 0
        while iteration < self.max_iterations:
            c = (a + b) / 2
            f_c = self.f(c, coefficients)
            error = abs(b - a) / 2

            self.iterations.append(
                {
                    "iteration": iteration + 1,
                    "a": a,
                    "b": b,
                    "c": c,
                    "f(c)": f_c,
                    "error": error,
                }
            )

            if abs(f_c) < self.tolerance or error < self.tolerance:
                return c, None

            if self.f(a, coefficients) * f_c < 0:
                b = c
            else:
                a = c

            iteration += 1

        return c, "Max iterations reached"


class RegulaFalsiMethod(RootFinder):
    """Regula Falsi (False Position) Method"""

    def solve(self, a, b, coefficients):
        self.iterations = []

        if self.f(a, coefficients) * self.f(b, coefficients) > 0:
            return None, "Error: f(a) and f(b) must have opposite signs"

        iteration = 0
        prev_c = a

        while iteration < self.max_iterations:
            f_a = self.f(a, coefficients)
            f_b = self.f(b, coefficients)

            c = a - (f_a * (b - a)) / (f_b - f_a)
            f_c = self.f(c, coefficients)
            error = abs(c - prev_c)

            self.iterations.append(
                {
                    "iteration": iteration + 1,
                    "a": a,
                    "b": b,
                    "c": c,
                    "f(c)": f_c,
                    "error": error,
                }
            )

            if abs(f_c) < self.tolerance or error < self.tolerance:
                return c, None

            if f_a * f_c < 0:
                b = c
            else:
                a = c

            prev_c = c
            iteration += 1

        return c, "Max iterations reached"


class SecantMethod(RootFinder):
    """Secant Method for root finding"""

    def solve(self, x0, x1, coefficients):
        self.iterations = []

        iteration = 0
        prev_x = x0

        while iteration < self.max_iterations:
            f_x0 = self.f(x0, coefficients)
            f_x1 = self.f(x1, coefficients)

            if abs(f_x1 - f_x0) < 1e-12:
                return None, "Denominator too small"

            x2 = x1 - (f_x1 * (x1 - x0)) / (f_x1 - f_x0)
            f_x2 = self.f(x2, coefficients)
            error = abs(x2 - x1)

            self.iterations.append(
                {
                    "iteration": iteration + 1,
                    "x0": x0,
                    "x1": x1,
                    "x2": x2,
                    "f(x2)": f_x2,
                    "error": error,
                }
            )

            if abs(f_x2) < self.tolerance or error < self.tolerance:
                return x2, None

            x0, x1 = x1, x2
            iteration += 1

        return x2, "Max iterations reached"


class NewtonRaphsonMethod(RootFinder):
    """Newton-Raphson Method"""

    def solve(self, x0, coefficients):
        self.iterations = []

        iteration = 0

        while iteration < self.max_iterations:
            f_x = self.f(x0, coefficients)
            df_x = self.df(x0, coefficients)

            if abs(df_x) < 1e-12:
                return None, "Derivative too close to zero"

            x1 = x0 - f_x / df_x
            error = abs(x1 - x0)

            self.iterations.append(
                {
                    "iteration": iteration + 1,
                    "x": x0,
                    "f(x)": f_x,
                    "f'(x)": df_x,
                    "x_new": x1,
                    "error": error,
                }
            )

            if abs(f_x) < self.tolerance or error < self.tolerance:
                return x1, None

            x0 = x1
            iteration += 1

        return x1, "Max iterations reached"


class FixedPointIterationMethod(RootFinder):
    """Fixed Point Iteration Method"""

    def solve(self, x0, coefficients, g_coefficients):
        self.iterations = []

        iteration = 0

        while iteration < self.max_iterations:
            f_x = self.f(x0, coefficients)
            g_x = self.f(x0, g_coefficients)
            error = abs(g_x - x0)

            self.iterations.append(
                {
                    "iteration": iteration + 1,
                    "x": x0,
                    "f(x)": f_x,
                    "g(x)": g_x,
                    "error": error,
                }
            )

            if abs(f_x) < self.tolerance or error < self.tolerance:
                return x0, None

            x0 = g_x
            iteration += 1

        return x0, "Max iterations reached"


class ModifiedSecantMethod(RootFinder):
    """Modified Secant Method (Perturbation Method)"""

    def solve(self, x0, h, coefficients):
        self.iterations = []

        iteration = 0

        while iteration < self.max_iterations:
            f_x = self.f(x0, coefficients)
            f_x_h = self.f(x0 + h, coefficients)

            denominator = (f_x_h - f_x) / h
            if abs(denominator) < 1e-12:
                return None, "Denominator too small"

            x1 = x0 - f_x / denominator
            error = abs(x1 - x0)

            self.iterations.append(
                {
                    "iteration": iteration + 1,
                    "x": x0,
                    "f(x)": f_x,
                    "x_new": x1,
                    "error": error,
                }
            )

            if abs(f_x) < self.tolerance or error < self.tolerance:
                return x1, None

            x0 = x1
            iteration += 1

        return x1, "Max iterations reached"


def print_menu():
    print("\n" + "=" * 60)
    print("ZERO OF FUNCTIONS (ZOF) SOLVER - CLI")
    print("=" * 60)
    print("Select a method:")
    print("1. Bisection Method")
    print("2. Regula Falsi (False Position) Method")
    print("3. Secant Method")
    print("4. Newton-Raphson Method")
    print("5. Fixed Point Iteration Method")
    print("6. Modified Secant Method")
    print("0. Exit")
    print("=" * 60)


def get_coefficients(
    prompt="Enter coefficients (a0, a1, a2, ...) for f(x)=a0+a1*x+a2*x^2+...:\n> ",
):
    while True:
        try:
            coeffs = list(map(float, input(prompt).split()))
            return coeffs
        except ValueError:
            print("Invalid input. Please enter numbers separated by spaces.")


def display_iterations(iterations):
    print("\n" + "-" * 80)
    print("ITERATION DETAILS:")
    print("-" * 80)

    if not iterations:
        print("No iterations completed.")
        return

    first_iter = iterations[0]
    headers = list(first_iter.keys())

    print(f"{'Iter':<6}", end="")
    for header in headers[1:]:
        print(f"{str(header):<15}", end="")
    print()
    print("-" * 80)

    for iteration_data in iterations:
        print(f"{iteration_data['iteration']:<6}", end="")
        for header in headers[1:]:
            value = iteration_data[header]
            if isinstance(value, float):
                print(f"{value:<15.6f}", end="")
            else:
                print(f"{str(value):<15}", end="")
        print()

    print("-" * 80)


def main():
    while True:
        print_menu()
        choice = input("Enter your choice (0-6): ").strip()

        if choice == "0":
            print("Exiting... Thank you for using ZOF Solver!")
            break

        try:
            tolerance = float(input("Enter tolerance (default 1e-6): ") or "1e-6")
            max_iter = int(input("Enter max iterations (default 100): ") or "100")

            if choice == "1":
                coefficients = get_coefficients()
                a = float(input("Enter lower bound a: "))
                b = float(input("Enter upper bound b: "))

                solver = BisectionMethod(tolerance, max_iter)
                root, error = solver.solve(a, b, coefficients)

                display_iterations(solver.iterations)
                if root is not None:
                    print(f"\nFinal Root: {root:.6f}")
                    print(f"Number of Iterations: {len(solver.iterations)}")
                    if solver.iterations:
                        print(f"Final Error: {solver.iterations[-1]['error']:.2e}")
                    print(
                        "Status: Converged successfully"
                        if not error
                        else f"Warning: {error}"
                    )
                else:
                    print("\nRoot: Not found")
                    if error:
                        print(f"Error: {error}")

            elif choice == "2":
                coefficients = get_coefficients()
                a = float(input("Enter lower bound a: "))
                b = float(input("Enter upper bound b: "))

                solver = RegulaFalsiMethod(tolerance, max_iter)
                root, error = solver.solve(a, b, coefficients)

                display_iterations(solver.iterations)
                if root is not None:
                    print(f"\nFinal Root: {root:.6f}")
                    print(f"Number of Iterations: {len(solver.iterations)}")
                    if solver.iterations:
                        print(f"Final Error: {solver.iterations[-1]['error']:.2e}")
                    print(
                        "Status: Converged successfully"
                        if not error
                        else f"Warning: {error}"
                    )
                else:
                    print("\nRoot: Not found")
                    if error:
                        print(f"Error: {error}")

            elif choice == "3":
                coefficients = get_coefficients()
                x0 = float(input("Enter first initial guess x0: "))
                x1 = float(input("Enter second initial guess x1: "))

                solver = SecantMethod(tolerance, max_iter)
                root, error = solver.solve(x0, x1, coefficients)

                display_iterations(solver.iterations)
                if root is not None:
                    print(f"\nFinal Root: {root:.6f}")
                    print(f"Number of Iterations: {len(solver.iterations)}")
                    if solver.iterations:
                        print(f"Final Error: {solver.iterations[-1]['error']:.2e}")
                    print(
                        "Status: Converged successfully"
                        if not error
                        else f"Warning: {error}"
                    )
                else:
                    print("\nRoot: Not found")
                    if error:
                        print(f"Error: {error}")

            elif choice == "4":
                coefficients = get_coefficients()
                x0 = float(input("Enter initial guess x0: "))

                solver = NewtonRaphsonMethod(tolerance, max_iter)
                root, error = solver.solve(x0, coefficients)

                display_iterations(solver.iterations)
                if root is not None:
                    print(f"\nFinal Root: {root:.6f}")
                    print(f"Number of Iterations: {len(solver.iterations)}")
                    if solver.iterations:
                        print(f"Final Error: {solver.iterations[-1]['error']:.2e}")
                    print(
                        "Status: Converged successfully"
                        if not error
                        else f"Warning: {error}"
                    )
                else:
                    print("\nRoot: Not found")
                    if error:
                        print(f"Error: {error}")

            elif choice == "5":
                coefficients = get_coefficients("Enter coefficients for f(x):\n> ")
                g_coefficients = get_coefficients("Enter coefficients for g(x):\n> ")
                x0 = float(input("Enter initial guess x0: "))

                solver = FixedPointIterationMethod(tolerance, max_iter)
                root, error = solver.solve(x0, coefficients, g_coefficients)

                display_iterations(solver.iterations)
                if root is not None:
                    print(f"\nFinal Root: {root:.6f}")
                    print(f"Number of Iterations: {len(solver.iterations)}")
                    if solver.iterations:
                        print(f"Final Error: {solver.iterations[-1]['error']:.2e}")
                    print(
                        "Status: Converged successfully"
                        if not error
                        else f"Warning: {error}"
                    )
                else:
                    print("\nRoot: Not found")
                    if error:
                        print(f"Error: {error}")

            elif choice == "6":
                coefficients = get_coefficients()
                x0 = float(input("Enter initial guess x0: "))
                h = float(input("Enter perturbation h (default 1e-6): ") or "1e-6")

                solver = ModifiedSecantMethod(tolerance, max_iter)
                root, error = solver.solve(x0, h, coefficients)

                display_iterations(solver.iterations)
                if root is not None:
                    print(f"\nFinal Root: {root:.6f}")
                    print(f"Number of Iterations: {len(solver.iterations)}")
                    if solver.iterations:
                        print(f"Final Error: {solver.iterations[-1]['error']:.2e}")
                    print(
                        "Status: Converged successfully"
                        if not error
                        else f"Warning: {error}"
                    )
                else:
                    print("\nRoot: Not found")
                    if error:
                        print(f"Error: {error}")

            else:
                print("Invalid choice. Please try again.")

        except ValueError as e:
            print(f"Error: Invalid input. {str(e)}")
        except Exception as e:
            print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
