"""
    InternalDataDrivenProblem

Preprocessed problem passed to data-driven algorithm implementations.

This type is developer API for solver packages. Application code should construct a
[`DataDrivenProblem`](@ref) and call `solve` instead.

# Fields

- `alg`: selected [`AbstractDataDrivenAlgorithm`](@ref).
- `testdata`: held-out data used to select a result.
- `traindata`: batches used to fit the result.
- `transform`: fitted data-normalization transform.
- `control_idx`: basis-to-control dependency indicators.
- `implicit_idx`: basis-to-implicit-variable dependency indicators.
- `parameter_idx`: basis entries that contain only parameters.
- `state_idx`: basis-to-state dependency indicators.
- `options`: shared [`DataDrivenCommonOptions`](@ref).
- `basis`: feature [`AbstractBasis`](@ref).
- `problem`: source [`AbstractDataDrivenProblem`](@ref).
- `kwargs`: algorithm-specific keyword arguments.
"""
struct InternalDataDrivenProblem{
        A <: AbstractDataDrivenAlgorithm, B <: AbstractBasis, TD,
        T <: DataLoader, F, CI, VI, PI, SI,
        O <: DataDrivenCommonOptions,
        P <: AbstractDataDrivenProblem, K,
    }
    # The Algorithm
    alg::A
    # Data and Normalization
    testdata::TD
    traindata::T
    transform::F
    # Indicators
    # Indicates which basis entries are dependent on controls
    control_idx::CI
    # Indicates which basis entries are dependent on implicit variables
    implicit_idx::VI
    # Indicates which basis entries are pure parameters
    parameter_idx::PI
    # Indicate which basis entries are dependent on the states
    state_idx::SI
    # Options
    options::O
    # Basis
    basis::B
    # The problem
    problem::P
    # Additional kwargs
    kwargs::K
end

"""
    get_fit_targets(alg, problem, basis) -> (inputs, targets)

Construct the matrices fitted by a data-driven algorithm.

The default evaluates `basis(problem)` and uses [`get_implicit_data`](@ref) as the
target. Algorithms whose target convention differs, such as Koopman algorithms, should
specialize this function.

# Arguments

- `alg::AbstractDataDrivenAlgorithm`: algorithm selecting the target convention.
- `problem::AbstractDataDrivenProblem`: source data.
- `basis::AbstractBasis`: feature basis evaluated on the source data.

# Returns

- `(inputs, targets)`: matrices passed to the algorithm implementation.
"""
function get_fit_targets(
        ::AbstractDataDrivenAlgorithm, prob::AbstractDataDrivenProblem,
        basis::AbstractBasis
    )
    Y = get_implicit_data(prob)
    X = basis(prob)
    return X, Y
end

# We always want a basis
function CommonSolve.init(
        prob::AbstractDataDrivenProblem, alg::AbstractDataDrivenAlgorithm;
        options::DataDrivenCommonOptions = DataDrivenCommonOptions(),
        kwargs...
    )
    return init(prob, unit_basis(prob), alg; options = options, kwargs...)
end

function CommonSolve.init(
        prob::AbstractDataDrivenProblem, basis::AbstractBasis,
        alg::AbstractDataDrivenAlgorithm = ZeroDataDrivenAlgorithm();
        options::DataDrivenCommonOptions = DataDrivenCommonOptions(),
        kwargs...
    )
    @unpack denoise, normalize, data_processing = options

    # This function handles preprocessing of the variables
    data = get_fit_targets(alg, prob, basis)

    if denoise
        optimal_shrinkage!(first(data))
    end

    # Get the information about structure
    control_idx = zeros(Bool, length(basis), length(controls(basis)))
    implicit_idx = zeros(Bool, length(basis), length(implicit_variables(basis)))
    state_idx = zeros(Bool, length(basis), length(states(basis)))
    parameter_idx = zeros(Bool, length(basis))

    for (i, eq) in enumerate(equations(basis))
        for (j, c) in enumerate(controls(basis))
            control_idx[i, j] = is_dependent(eq.rhs, unwrap(c))
        end
        for (k, v) in enumerate(implicit_variables(basis))
            implicit_idx[i, k] = is_dependent(eq.rhs, unwrap(v))
        end
        for (k, v) in enumerate(states(basis))
            state_idx[i, k] = is_dependent(eq.rhs, unwrap(v))
        end

        parameter_idx[i] = all(ModelingToolkitBase.isparameter, Symbolics.get_variables(eq.rhs))
    end

    # We do not center, given that we can have constants in our Basis!
    dt = fit(normalize, first(data))

    apply_transform!(dt, first(data))

    test, loader = data_processing(data)

    return InternalDataDrivenProblem(
        alg, test, loader, dt, control_idx, implicit_idx,
        parameter_idx, state_idx,
        options, basis, prob, kwargs
    )
end

function CommonSolve.solve!(::InternalDataDrivenProblem{ZeroDataDrivenAlgorithm})
    @warn "No sufficient algorithm chosen! Return ErrorDataDrivenResult!"
    return ErrorDataDrivenResult()
end
