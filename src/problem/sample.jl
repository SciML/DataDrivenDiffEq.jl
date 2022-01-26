function train_test(n_obs, split, start = 1)
    @assert 0 < split < 1 "Split value has to be in (0,1)." 
    train = floor(Int, split*n_obs)
    test = (start).+(train:n_obs)
    train = (start-1).+(1:train)
    train, test
end

function data_split(n_obs, nsplits)
    @assert 0 < nsplits < n_obs-1 "Number of splits has to be in (0, n-1)."
    batchsize = floor(Int, n_obs/(nsplits+1))
    start = 1
    map(1:nsplits) do n
        cs = start:(n == nsplits ? n_obs : start+batchsize)
        start += batchsize+1
        cs
    end
end

function subsample(n_obs, freq, start = 1)
    @assert 0 < freq < n_obs/2 "Subsampling frequency in (0, n/2)."
    start:freq:n_obs 
end

function subsample(t::AbstractVector{T}, period::T) where T <: Real
    @assert period > zero(typeof(period)) "Sampling period has to be positive."
    @assert t[end]-t[1]>= period "Subsampling impossible. Sampling period exceeds time window."
    idx = Int64[1]
    t_now = t[1]
    @inbounds for (i, t_current) in enumerate(t)
        if t_current  >= period -5*eps() + t_now
            push!(idx, i)
            t_now = t_current
        end
    end
    return idx
end

function train_test_split!(prob::AbstractDataDrivenProblem, split)
    
end