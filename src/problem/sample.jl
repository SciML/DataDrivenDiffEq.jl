abstract type AbstractSampler end

"""
$(TYPEDEF)

A simple sampler container. 
Takes in `AbstractSampler`s to apply onto a `DataDrivenProblem` in the order they are given.
If a `Split` sampler is provided, then it will be moved to the first index by definition. 
"""
struct DataSampler{T} <: AbstractSampler
    samplers::T

    function DataSampler(args...)
        # Check if there is a split and move this to the front, 
        # Delete all other
        has_split = map(x -> isa(x, Split), args)
        if any(has_split)
            idx = findfirst(has_split)
            samplers = [x for (i, x) in enumerate(args) if !has_split[i]]
            t = tuple(args[idx], samplers...)
        else
            t = args
        end
        return new{typeof(t)}(t)
    end
end

DataSampler() = DataSampler(Split(ratio = 1.0))

(s::DataSampler)(p::AbstractDataDrivenProblem) = begin
    isempty(s.samplers) && return ([1:length(p)], :)
    train, test = ∘(reverse(s.samplers)...)(p)
    train = isa(train, AbstractRange) ? [train] : train
    test = isnothing(test) ? Colon() : test
    return train, test
end

"""
$(TYPEDEF)

Performs a train test split of the `DataDrivenProblem` where `ratio` 
defines the (rough) percentage of training data. 

The optional keyword `shuffle` indicates to sample from random shuffles of the data, allowing 
for repetition.

Returns ranges for training and testing data.
"""
@with_kw struct Split <: AbstractSampler
    ratio::Real = 0.8
    shuffle::Bool = false
end

function (s::Split)(p::AbstractDataDrivenProblem, rng = Random.GLOBAL_RNG)
    n_obs = length(p)
    return s((1:n_obs, nothing), rng)
end

function (s::Split)(data::Tuple, rng = Random.GLOBAL_RNG)
    train, test = data

    @unpack ratio, shuffle = s
    (ratio <= 0 || ratio >= 1) && return (train, train)

    n_obs = length(train)

    train_ = floor(Int, ratio * n_obs)

    if !shuffle
        test = (1+train_):n_obs
        train = 1:train_
    else
        idx = randperm(rng, n_obs)
        train = idx[1:train_]
        test = idx[train_+1:end]
    end
    train, test
end

"""
$(TYPEDEF)

Partitions the `DataDrivenProblem` into `n` equal partitions. If used after performing a train test `Split`, works just on the training data.

The optional keyword `shuffle` indicates to sample from random shuffles of the data, allowing 
for repetition.

The optional keyword `repeated` indicates to allow for repeated sampling of data points.

`batchsize_min` is the minimum batchsize, which should be used within each partition of the dataset.

Returns ranges for each partition of the provided data.
"""
@with_kw struct Batcher <: AbstractSampler
    """Number of partitions"""
    n::Int = 1
    """Minimum Batchsize"""
    batchsize_min::Int = 0
    """Shuffle the data before sampling"""
    shuffle::Bool = false
    """Allow intersecting datasets"""
    repeated::Bool = false
end

function (p::Batcher)(prob::AbstractDataDrivenProblem, rng = Random.GLOBAL_RNG)
    return p((1:length(prob), nothing), rng)
end

function (p::Batcher)(data, rng = Random.GLOBAL_RNG)
    training, test = data
    @unpack n, shuffle, repeated, batchsize_min = p
    n <= 1 && return training, test
    n_obs = length(training)

    @assert n < n_obs "Number of splits has to be less than data size."

    batchsize = floor(Int, n_obs / n - 1)
    batchsize = max(batchsize, batchsize_min)

    if !shuffle && !repeated
        start = first(training)

        return map(1:n) do n_
            cs = start:(n_ == n ? n_obs : start + batchsize)
            start += batchsize + 1
            training[cs]
        end, test

    elseif !repeated
        idx = randperm(rng, n_obs)
        start = first(training)

        return map(1:n) do n_
            cs = start:(n_ == n ? n_obs : start + batchsize)
            start += batchsize + 1
            idx[cs]
        end, test
    else
        s = 0
        return map(1:n) do n_
            idx = shuffle ? randperm(rng, n_obs) : training
            if n_ == n
                sample(rng, idx, n_obs - s, replace = false)
            else
                s += batchsize + 1
                sample(rng, idx, batchsize + 1, replace = false)
            end
        end, test
    end
end

