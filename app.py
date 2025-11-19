from abc import ABC, abstractmethod

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Set page config
st.set_page_config(
    page_title="ZOF Solver",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 16px;
    }
    </style>
""",
    unsafe_allow_html=True,
)


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
                    "Iteration": iteration + 1,
                    "a": a,
                    "b": b,
                    "c (root)": c,
                    "f(c)": f_c,
                    "Error": error,
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
                    "Iteration": iteration + 1,
                    "a": a,
                    "b": b,
                    "c (root)": c,
                    "f(c)": f_c,
                    "Error": error,
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
    def solve(self, x0, x1, coefficients):
        self.iterations = []
        iteration = 0

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
                    "Iteration": iteration + 1,
                    "x0": x0,
                    "x1": x1,
                    "x2 (root)": x2,
                    "f(x2)": f_x2,
                    "Error": error,
                }
            )

            if abs(f_x2) < self.tolerance or error < self.tolerance:
                return x2, None

            x0, x1 = x1, x2
            iteration += 1

        return x2, "Max iterations reached"


class NewtonRaphsonMethod(RootFinder):
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
                    "Iteration": iteration + 1,
                    "x": x0,
                    "f(x)": f_x,
                    "f'(x)": df_x,
                    "x_new": x1,
                    "Error": error,
                }
            )

            if abs(f_x) < self.tolerance or error < self.tolerance:
                return x1, None

            x0 = x1
            iteration += 1

        return x1, "Max iterations reached"


class FixedPointIterationMethod(RootFinder):
    def solve(self, x0, coefficients, g_coefficients):
        self.iterations = []
        iteration = 0

        while iteration < self.max_iterations:
            f_x = self.f(x0, coefficients)
            g_x = self.f(x0, g_coefficients)
            error = abs(g_x - x0)

            self.iterations.append(
                {
                    "Iteration": iteration + 1,
                    "x": x0,
                    "f(x)": f_x,
                    "g(x)": g_x,
                    "Error": error,
                }
            )

            if abs(f_x) < self.tolerance or error < self.tolerance:
                return g_x, None

            x0 = g_x
            iteration += 1

        return x0, "Max iterations reached"


class ModifiedSecantMethod(RootFinder):
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
                    "Iteration": iteration + 1,
                    "x": x0,
                    "f(x)": f_x,
                    "x_new": x1,
                    "Error": error,
                }
            )

            if abs(f_x) < self.tolerance or error < self.tolerance:
                return x1, None

            x0 = x1
            iteration += 1

        return x1, "Max iterations reached"


# Streamlit UI
st.title("📊 Zero of Functions (ZOF) Solver")
st.markdown("---")

col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("⚙️ Configuration")

    method = st.selectbox(
        "Select Method",
        [
            "Bisection",
            "Regula Falsi",
            "Secant",
            "Newton-Raphson",
            "Fixed Point Iteration",
            "Modified Secant",
        ],
    )

    tolerance = st.number_input("Tolerance", value=1e-6, format="%.2e")
    max_iterations = st.number_input("Max Iterations", value=100, step=1)

with col2:
    st.subheader("📝 Equation Input")
    st.markdown("Enter polynomial coefficients for: **f(x) = a₀ + a₁x + a₂x² + ...**")
    coeffs_input = st.text_input("Coefficients (space-separated)", "-1 0 1")

try:
    coefficients = list(map(float, coeffs_input.split()))
except:
    st.error("Invalid coefficients format")
    st.stop()

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    if method == "Bisection":
        a = st.number_input("Lower bound (a)", value=-2.0)
        b = st.number_input("Upper bound (b)", value=2.0)

        if st.button("🔍 Solve", key="bisection"):
            solver = BisectionMethod(tolerance, max_iterations)
            root, error = solver.solve(a, b, coefficients)
            st.session_state.result = (root, error, solver.iterations)

    elif method == "Regula Falsi":
        a = st.number_input("Lower bound (a)", value=-2.0)
        b = st.number_input("Upper bound (b)", value=2.0)

        if st.button("🔍 Solve", key="falsi"):
            solver = RegulaFalsiMethod(tolerance, max_iterations)
            root, error = solver.solve(a, b, coefficients)
            st.session_state.result = (root, error, solver.iterations)

    elif method == "Secant":
        x0 = st.number_input("Initial guess x₀", value=-1.0)
        x1 = st.number_input("Initial guess x₁", value=1.0)

        if st.button("🔍 Solve", key="secant"):
            solver = SecantMethod(tolerance, max_iterations)
            root, error = solver.solve(x0, x1, coefficients)
            st.session_state.result = (root, error, solver.iterations)

    elif method == "Newton-Raphson":
        x0 = st.number_input("Initial guess x₀", value=1.0)

        if st.button("🔍 Solve", key="newton"):
            solver = NewtonRaphsonMethod(tolerance, max_iterations)
            root, error = solver.solve(x0, coefficients)
            st.session_state.result = (root, error, solver.iterations)

    elif method == "Fixed Point Iteration":
        st.markdown("**f(x) coefficients:**")
        coeffs_f = st.text_input("f(x) coefficients", "-1 0 1", key="f_coeffs")
        st.markdown("**g(x) coefficients:**")
        coeffs_g = st.text_input("g(x) coefficients", "0 0.5 0.5", key="g_coeffs")
        x0 = st.number_input("Initial guess x₀", value=0.5)

        if st.button("🔍 Solve", key="fixed"):
            try:
                coeff_f = list(map(float, coeffs_f.split()))
                coeff_g = list(map(float, coeffs_g.split()))
                solver = FixedPointIterationMethod(tolerance, max_iterations)
                root, error = solver.solve(x0, coeff_f, coeff_g)
                st.session_state.result = (root, error, solver.iterations)
            except:
                st.error("Invalid coefficients")

    elif method == "Modified Secant":
        x0 = st.number_input("Initial guess x₀", value=1.0)
        h = st.number_input("Perturbation h", value=1e-6, format="%.2e")

        if st.button("🔍 Solve", key="mod_secant"):
            solver = ModifiedSecantMethod(tolerance, max_iterations)
            root, error = solver.solve(x0, h, coefficients)
            st.session_state.result = (root, error, solver.iterations)

if "result" in st.session_state:
    with col2:
        root, error, iterations = st.session_state.result

        if root is not None:
            st.success("✅ Solution Found!")
            st.metric("Root", f"{root:.8f}")
            st.metric("Iterations", len(iterations))
            if iterations:
                st.metric("Final Error", f"{iterations[-1]['Error']:.2e}")
        else:
            st.error(f"❌ {error}")

st.markdown("---")
st.subheader("📊 Results")

if "result" in st.session_state:
    root, error, iterations = st.session_state.result

    if iterations:
        df = pd.DataFrame(iterations)
        st.dataframe(df, use_container_width=True)

        # Plot convergence
        if "Error" in df.columns:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=df["Iteration"],
                    y=df["Error"],
                    mode="lines+markers",
                    name="Error",
                    line=dict(color="red", width=2),
                )
            )
            fig.update_layout(
                title="Convergence Rate",
                xaxis_title="Iteration",
                yaxis_title="Error",
                yaxis_type="log",
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)
