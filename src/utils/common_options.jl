
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
    """Train test split"""
    split::Real = 0.8
    """Shuffle the training data"""
    shuffle::Bool = false
    """Batchsizes to use, if zero no batching is performed"""
    batchsize::Int = 0
    """Using partial batches"""
    partial::Bool = true
    """Random seed"""
    rng::Random.AbstractRNG = Random.default_rng()
end

function (d::DataProcessing)(X,Y)
    @unpack split, shuffle, batchsize, partial, rng = d
    
    split = split ∈ (0, 1) ? split : max(0., min(split, 1.))
    
    batchsize = batchsize < 1 ? 1 : batchsize

    xtrain, xtest = splitobs((X, Y), at = split, shuffle = false)
    
    xtest, DataLoader(
        xtrain, batchsize = batchsize, partial = partial, shuffle = true, rng = rng
    )
end

"""
$(TYPEDEF)

A wrapper to normalize the data using `StatsBase.jl`. Performs normalization over the full problem data
given the type of the normalization (`Nothing`, `ZScoreTransform`, `UnitRangeTransform`).

If no `nothing` is used, no normalization is performed.

## Note

Given that `DataDrivenDiffEq.jl` allows for constants in the basis, the `center` keyword of `StatsBase.fit` is set to false.
"""
struct DataNormalization{T <: Union{Nothing, ZScoreTransform, UnitRangeTransform}}
    method::T 
end

DataNormalization() = DataNormalization(nothing)

StatsBase.fit(::DataNormalization{Nothing}, data) = fit(ZScoreTransform, data, dims = 2, scale = false, center = false)
StatsBase.fit(::DataNormalization{T}, data) where T = fit(T, data; dims = 2, center = false)


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
    eval_expresssion::Bool = true
    """Additional kwargs"""
    kwargs::K = (;)
end

## INTERNAL USE FOR PREPROCESSING

# This is a way to create a datadriven problem relatively efficient.
struct InternalDataDrivenProblem{B <: AbstractBasis, TD, T <: DataLoader, F, O <: DataDrivenCommonOptions, P <: AbstractDataDrivenProblem}
    testdata::TD
    traindata::T
    transform::F
    options::O
    basis::B
    problem::P
end

# We always want a basis!
__preprocess(prob::AbstractDataDrivenProblem, options::DataDrivenCommonOptions) = __preprocess(prob, unit_basis(prob), options)

function __preprocess(prob::AbstractDataDrivenProblem, basis::AbstractBasis , options::DataDrivenCommonOptions = DataDrivenCommonOptions())
    @unpack denoise, normalize, data_processing = options

    Θ = basis(prob)
    Y = get_implicit_data(prob)

    if denoise 
        optimal_shrinkage!(Θ)
    end

    # We do not center, given that we can have constants in our Basis!
    dt = fit(normalize, Θ)
    
    StatsBase.transform!(dt, Θ)
    
    test, loader =  data_processing(Θ, Y)

    return InternalDataDrivenProblem(
        test, loader, dt, options, basis, prob
    )
end

