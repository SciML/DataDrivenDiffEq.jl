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
    return xtest,
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

function StatsAPI.fit(::DataNormalization{Nothing}, data)
    return fit(ZScoreTransform, data, dims = 2, scale = false, center = false)
end

function StatsAPI.fit(::DataNormalization{UnitRangeTransform}, data)
    tf = fit(UnitRangeTransform, data, dims = 2)
    # Adapt for constants here
    tf.scale .= [isinf(s) ? one(eltype(s)) : s for s in tf.scale]
    return tf
end

function StatsAPI.fit(::DataNormalization{ZScoreTransform}, data)
    tf = fit(ZScoreTransform, data, dims = 2, center = false)
    # Adapt for constants here
    tf.scale .= [iszero(s) ? one(eltype(s)) : s for s in tf.scale]
    return tf
end

"""
    apply_transform(transform, data) -> transformed_data

Apply a fitted data-normalization transform and return a transformed copy of `data`.

This is developer API for DataDrivenDiffEq solver packages. The supported transforms are
`StatsBase.ZScoreTransform` and `StatsBase.UnitRangeTransform`, which are the transforms
produced by [`DataNormalization`](@ref).

# Arguments

- `transform`: fitted normalization transform.
- `data::AbstractArray`: numeric data arranged consistently with the fitted transform.

# Returns

- `transformed_data`: a transformed copy of `data`.
"""
apply_transform(transform, data) = apply_transform!(transform, copy(data))

"""
    apply_transform!(transform, data) -> data

Apply a fitted data-normalization transform to `data` in place.

This is the mutating form of [`apply_transform`](@ref) and is developer API for
DataDrivenDiffEq solver packages.
"""
function apply_transform!(transform::ZScoreTransform, data::AbstractMatrix{<:Real})
    offset = transform.dims == 1 ? reshape(transform.mean, 1, :) : reshape(transform.mean, :, 1)
    scale = transform.dims == 1 ? reshape(transform.scale, 1, :) : reshape(transform.scale, :, 1)

    isempty(offset) || (data .-= offset)
    isempty(scale) || (data ./= scale)
    return data
end

function apply_transform!(transform::UnitRangeTransform, data::AbstractMatrix{<:Real})
    offset = transform.dims == 1 ? reshape(transform.min, 1, :) : reshape(transform.min, :, 1)
    scale = transform.dims == 1 ? reshape(transform.scale, 1, :) : reshape(transform.scale, :, 1)

    transform.unit && (data .-= offset)
    data .*= scale
    return data
end

function apply_transform!(transform, data::AbstractVector{<:Real})
    apply_transform!(transform, reshape(data, :, 1))
    return data
end
