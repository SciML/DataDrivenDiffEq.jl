"""
$(TYPEDEF)

Defines a preprocessing pipeline for the data using `MLUtils.jl`.
All of the fields can be set using keyword arguments.

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
    """Batch size to use, if zero no batching is performed"""
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

## Normalization

"""
$(TYPEDEF)

A wrapper to normalize the data using `StatsBase.jl`. Performs normalization over the full problem data
given the type of the normalization (`Nothing`, `ZScoreTransform`, `UnitRangeTransform`).

If no `nothing` is used, no normalization is performed.

## Note

Given that `DataDrivenDiffEq.jl` allows for constants in the basis, the `center` keyword of `StatsBase.fit` is set to false.
Additionally, constants will be scaled with `1`.
"""
struct DataNormalization{T <: Union{Nothing, ZScoreTransform, UnitRangeTransform}}
end

DataNormalization() = DataNormalization{Nothing}()
DataNormalization(method::Type{T}) where {T} = DataNormalization{T}()

function StatsBase.fit(::DataNormalization{Nothing}, data)
    StatsBase.fit(ZScoreTransform, data, dims = 2, scale = false, center = false)
end

function StatsBase.fit(::DataNormalization{UnitRangeTransform}, data)
    tf = StatsBase.fit(UnitRangeTransform, data, dims = 2)
    # Adapt for constants here
    tf.scale .= [isinf(s) ? one(eltype(s)) : s for s in tf.scale]
    tf
end

function StatsBase.fit(::DataNormalization{ZScoreTransform}, data)
    tf = StatsBase.fit(ZScoreTransform, data, dims = 2, center = false)
    # Adapt for constants here
    tf.scale .= [iszero(s) ? one(eltype(s)) : s for s in tf.scale]
    tf
end
