# Precompilation workload for DataDrivenDiffEq
# This file contains code that will be executed during precompilation to
# improve startup time and time-to-first-execution (TTFX)

using PrecompileTools: @compile_workload, @setup_workload

@setup_workload begin
    # Set up minimal test data and variables
    # Note: We avoid heavy computations here, just trigger code paths
    @compile_workload begin
        # Create symbolic variables for Basis construction
        # This is the main TTFX bottleneck (33+ seconds without precompilation)
        @variables x y t

        # Precompile Basis construction - the most important operation
        # Use simple expressions to minimize precompilation time
        basis = Basis([x, y, x * y, x^2, y^2], [x, y])

        # Precompile basis evaluation (both scalar and vector forms)
        u_test = [1.0, 2.0]
        p_test = Float64[]
        t_test = 0.0
        _ = basis(u_test, p_test, t_test)

        # Precompile problem constructors
        X = randn(2, 10)
        Y = randn(2, 10)
        t_span = collect(range(0.0, 1.0, length = 10))

        # DirectDataDrivenProblem
        prob_direct = DirectDataDrivenProblem(X, Y)

        # DiscreteDataDrivenProblem
        prob_discrete = DiscreteDataDrivenProblem(X)

        # ContinuousDataDrivenProblem (uses collocation which takes time)
        prob_cont = ContinuousDataDrivenProblem(X, t_span)

        # Precompile basis generators
        poly_basis = polynomial_basis([x, y], 2)
        mono_basis = monomial_basis([x, y], 2)

        # Precompile basis evaluation with problem
        _ = basis(prob_direct)
    end
end
