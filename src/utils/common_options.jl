
"""
$(TYPEDEF)

Defines a preprocessing pipeline for the data using `MLUtils.jl`. 
All of the fields can be set using keyworded arguments.

# Fields
$(FIELDS)

## Note 

Currently, only `splitobs` for a train-test split and `DataLoader` is wrapped.
Other algorithms may follow. 
"""
@with_kw struct DataProcessing
    """Train test split, indicates the (rough) percentage of training data"""
    split::Real = 1.0
    """Shuffle the training data"""
    shuffle::Bool = false
    """Batchsizes to use, if zero no batching is performed"""
    batchsize::Int = 0
    """Using partial batches"""
    partial::Bool = true
    """Random seed"""
    rng::Random.AbstractRNG = Random.default_rng()
end

function (d::DataProcessing)(data::Tuple)
    @unpack split, shuffle, batchsize, partial, rng = d
    X = first(data)
    split = (0.0 <= split <= 1.0) ? split : max(0.0, min(split, 1.0))

    xtrain, xtest = splitobs(data, at = split, shuffle = false)

    batchsize = batchsize <= 0 ? size(first(xtrain), 2) : batchsize
    xtest,
    DataLoader(xtrain, batchsize = batchsize, partial = partial, shuffle = true, rng = rng)
end

(d::DataProcessing)(X, Y) = d((X, Y))

"""
$(TYPEDEF)

A wrapper to normalize the data using `StatsBase.jl`. Performs normalization over the full problem data
given the type of the normalization (`Nothing`, `ZScoreTransform`, `UnitRangeTransform`).

If no `nothing` is used, no normalization is performed.

## Note

Given that `DataDrivenDiffEq.jl` allows for constants in the basis, the `center` keyword of `StatsBase.fit` is set to false.
"""
struct DataNormalization{T <: Union{Nothing, ZScoreTransform, UnitRangeTransform}}
end

DataNormalization() = DataNormalization{Nothing}()
DataNormalization(method::Type{T}) where {T} = DataNormalization{T}()

function StatsBase.fit(::DataNormalization{Nothing}, data)
    StatsBase.fit(ZScoreTransform, data, dims = 2, scale = false, center = false)
end
function StatsBase.fit(::DataNormalization{UnitRangeTransform}, data)
    StatsBase.fit(UnitRangeTransform, data, dims = 2)
end
function StatsBase.fit(::DataNormalization{ZScoreTransform}, data) where {T}
    StatsBase.fit(ZScoreTransform, data, dims = 2, center = false)
end

"""
$(TYPEDEF)

Common options for all methods provided via `DataDrivenDiffEq`. 

# Fields
$(FIELDS)
    
## Note

The keyword argument `eval_expression` controls the function creation
behavior. `eval_expression=true` means that `eval` is used, so normal
world-age behavior applies (i.e. the functions cannot be called from
the function that generates them). If `eval_expression=false`,
then construction via GeneralizedGenerated.jl is utilized to allow for
same world-age evaluation. However, this can cause Julia to segfault
on sufficiently large basis functions. By default eval_expression=false.

Denoising happens before normalization!
"""
@with_kw struct DataDrivenCommonOptions{T, K}
    # Optimization options
    """Maximum iterations"""
    maxiters::Int = 1_00
    """Absolute tolerance"""
    abstol::T = sqrt(eps())
    """Relative tolerance"""
    reltol::T = sqrt(eps())
    """Show a progress meter"""
    progress::Bool = false
    """Display log - Not implemented right now"""
    verbose::Bool = false
    # Preprocessing
    """Denoise the data using the [`optimal threshold`](https://arxiv.org/abs/1305.5870) method."""
    denoise::Bool = false
    """Normalize the data, see [`DataNormalization`](@ref)"""
    normalize::DataNormalization = DataNormalization()
    """Data processing pipeline, see [`DataProcessing`](@ref)"""
    data_processing::DataProcessing = DataProcessing()
    # Postprocessing
    """Significant digits for the parameters - used for rounding. Default = 10"""
    digits::Int = 10
    """Evaluate the expression, see [`Symbolics.build_function`](https://symbolics.juliasymbolics.org/stable/manual/build_function/)"""
    eval_expresssion::Bool = false
    """Additional kwargs"""
    kwargs::K = (;)
end

## INTERNAL USE ONLY

# This is a way to create a datadriven problem relatively efficient and handle all algorithms
struct InternalDataDrivenProblem{A <: AbstractDataDrivenAlgorithm, B <: AbstractBasis, TD,
                                 T <: DataLoader, F, CI, VI, O <: DataDrivenCommonOptions,
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

    for (i, eq) in enumerate(equations(basis))
        for (j, c) in enumerate(controls(basis))
            control_idx[i, j] = is_dependent(eq.rhs, Symbolics.unwrap(c))
        end
        for (k, v) in enumerate(implicit_variables(basis))
            implicit_idx[i, k] = is_dependent(eq.rhs, Symbolics.unwrap(v))
        end
    end

    # We do not center, given that we can have constants in our Basis!
    dt = fit(normalize, first(data))

    StatsBase.transform!(dt, first(data))

    test, loader = data_processing(data)

    return InternalDataDrivenProblem(alg, test, loader, dt, control_idx, implicit_idx,
                                     options, basis, prob, kwargs)
end

function CommonSolve.solve!(::InternalDataDrivenProblem{ZeroDataDrivenAlgorithm})
    @warn "No sufficient algorithm choosen! Return ErrorDataDrivenResult!"
    return ErrorDataDrivenResult()
end
