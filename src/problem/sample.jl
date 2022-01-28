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
        has_split = map(x->isa(x, Split), args)
        idx = findfirst(has_split)
        samplers = [x for (i,x) in enumerate(args) if !has_split[i]]
        t =  tuple(args[idx], samplers...)
        return new{typeof(t)}(t)
    end
end

DataSampler() = DataSampler(Split(ratio = 0.8))
(s::DataSampler)(p::AbstractDataDrivenProblem) = ∘(reverse(s.samplers)...)(p)

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
    return s((1:n_obs,nothing), rng)
end

function (s::Split)(data::Tuple, rng = Random.GLOBAL_RNG)
    train, test = data
    
    @unpack ratio, shuffle = s
    @assert 0 < ratio < 1 "Split value has to be in (0,1)." 
   
    n_obs = length(train)

    train_ = floor(Int, ratio*n_obs)

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

Returns ranges for each partition of the provided data.
"""
@with_kw struct Partition <: AbstractSampler 
    n::Int = 1
    shuffle::Bool = false
end

function (p::Partition)(prob::AbstractDataDrivenProblem, rng = Random.GLOBAL_RNG)
    return p((1:length(prob),nothing), rng)
end

function (p::Partition)(data, rng = Random.GLOBAL_RNG)
    training, test = data
    @unpack n, shuffle = p
    n <= 1 && return training
    n_obs = length(training)

    @assert n < n_obs "Number of splits has to be less than data size."

    batchsize = floor(Int, n_obs/(n+1))

    if !shuffle
        start = first(training)

        return map(1:n) do n_
            cs = start:(n_ == n ? n_obs : start+batchsize)
            start += batchsize+1
            training[cs]
        end, test
    else
        idx = randperm(rng, n_obs)
        
        return map(1:n) do n_
            idx[(n_-1)*batchsize+1:n_*batchsize]
        end, test
    end
end


#function subsample(n_obs, freq, start = 1)
#    @assert 0 < freq < n_obs/2 "Subsampling frequency in (0, n/2)."
#    start:freq:n_obs 
#end

#function subsample(t::AbstractVector{T}, period::T) where T <: Real
#    @assert period > zero(typeof(period)) "Sampling period has to be positive."
#    @assert t[end]-t[1]>= period "Subsampling impossible. Sampling period exceeds time window."
#    idx = Int64[1]
#    t_now = t[1]
#    @inbounds for (i, t_current) in enumerate(t)
#        if t_current  >= period -5*eps() + t_now
#            push!(idx, i)
#            t_now = t_current
#        end
#    end
#    return idx
#end

