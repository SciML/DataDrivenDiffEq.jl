
## INTERNAL USE ONLY

# This is a way to create a datadriven problem relatively efficient and handle all algorithms
struct InternalDataDrivenProblem{A <: AbstractDataDrivenAlgorithm, B <: AbstractBasis, TD,
                                 T <: DataLoader, F, CI, VI, PI, SI,
                                 O <: DataDrivenCommonOptions,
                                 P <: AbstractDataDrivenProblem, K}
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

# This is a preprocess step, which commonly returns the implicit data.
# For Koopman Algorithms this is not true
function get_fit_targets(::AbstractDataDrivenAlgorithm, prob::AbstractDataDrivenProblem,
                         basis::AbstractBasis)
    Y = get_implicit_data(prob)
    X = basis(prob)
    return X, Y
end

# We always want a basis
function CommonSolve.init(prob::AbstractDataDrivenProblem, alg::AbstractDataDrivenAlgorithm;
                          options::DataDrivenCommonOptions = DataDrivenCommonOptions(),
                          kwargs...)
    init(prob, unit_basis(prob), alg; options = options, kwargs...)
end

function CommonSolve.init(prob::AbstractDataDrivenProblem, basis::AbstractBasis,
                          alg::AbstractDataDrivenAlgorithm = ZeroDataDrivenAlgorithm();
                          options::DataDrivenCommonOptions = DataDrivenCommonOptions(),
                          kwargs...)
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
            control_idx[i, j] = is_dependent(eq.rhs, Symbolics.unwrap(c))
        end
        for (k, v) in enumerate(implicit_variables(basis))
            implicit_idx[i, k] = is_dependent(eq.rhs, Symbolics.unwrap(v))
        end
        for (k, v) in enumerate(states(basis))
            state_idx[i, k] = is_dependent(eq.rhs, Symbolics.unwrap(v))
        end

        parameter_idx[i] = all(ModelingToolkit.isparameter, Symbolics.get_variables(eq.rhs))
    end

    # We do not center, given that we can have constants in our Basis!
    dt = fit(normalize, first(data))

    StatsBase.transform!(dt, first(data))

    test, loader = data_processing(data)

    return InternalDataDrivenProblem(alg, test, loader, dt, control_idx, implicit_idx,
                                     parameter_idx, state_idx,
                                     options, basis, prob, kwargs)
end

function CommonSolve.solve!(::InternalDataDrivenProblem{ZeroDataDrivenAlgorithm})
    @warn "No sufficient algorithm choosen! Return ErrorDataDrivenResult!"
    return ErrorDataDrivenResult()
end
