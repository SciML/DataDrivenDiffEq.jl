struct PathStatistics{T} <: StatsBase.StatisticalModel
    rss::T
    loglikelihood::T
    nullloglikelihood::T
    dof::Int
    nobs::Int
end

function update_stats!(stats::PathStatistics{T}, rss::T, ll::T, dof::Int)
    @set! stats.dof = dof
    @set! stats.loglikelihood = ll
    @set! stats.rss = rss
    return
end

StatsBase.rss(stats::PathStatistics) = getfield(stats, :rss)
StatsBase.nobs(stats::PathStatistics) = getfield(stats, :nobs)
StatsBase.loglikelihood(stats::PathStatistics) = getfield(stats, :loglikelihood)
StatsBase.nullloglikelihood(stats::PathStatistics) = getfield(stats, :nullloglikelihood)
StatsBase.dof(stats::PathStatistics) = getfield(stats, :dof)
StatsBase.r2(c::Candidate) = r2(c, :CoxSnell)


struct ComponentModel{B, M}
    basis::B
    model::M
end

(c::ComponentModel)(dataset::Dataset{T}, ps, st::NamedTuple{fieldnames}, p::AbstractVector{T}) where {T, fieldnames} = first(c.model(c.basis(dataset, p), ps, st))
(c::ComponentModel)(ps, st::NamedTuple{fieldnames}, paths::Vector{<:AbstractPathState}) where {fieldnames} = get_loglikelihood(c.model, ps, st, paths)

"""
$(TYPEDEF)

A container holding all the information for the current candidate solution
to the symbolic regression problem.

# Fields
$(FIELDS)
"""
struct Candidate{S <: NamedTuple} <: StatsBase.StatisticalModel
    "Random seed"
    rng::Random.AbstractRNG
    "The current state"
    st::S
    "The current parameters"
    ps::AbstractVector
    "Incoming paths"
    incoming_path::Vector{AbstractPathState}
    "Outgoing path"
    outgoing_path::Vector{AbstractPathState}
    "Statistics"
    statistics::PathStatistics
    "The observed model"
    observed::ObservedModel
    "The parameter distribution"
    parameterdist::ParameterDistributions
    "The optimal scales"
    scales::AbstractVector
    "The optimal parameters"
    parameters::AbstractVector
    "The component model"
    model::ComponentModel
end

(c::Candidate)(dataset::Dataset{T}, ps = c.ps, p = c.parameters) = c.model(dataset, ps, c.st, transform_parameters(c.parameterdist, p)) 
(c::Candidate)(ps = c.ps) = c.model(ps, c.st, c.outgoing_path)

Base.print(io::IO, c::Candidate) = print(io, "Candidate $(rss(c))")
Base.show(io::IO, c::Candidate) = print(io, c)
Base.summary(io::IO, c::Candidate) = print(io, c)

StatsBase.rss(c::Candidate) = sum(rss, c.statistics)
StatsBase.nobs(c::Candidate) = sum(nobs, c.statistics)
StatsBase.loglikelihood(c::Candidate) = sum(loglikelihood, c.statistics)
StatsBase.nullloglikelihood(c::Candidate) = sum(nullloglikelihood, c.statistics)
StatsBase.dof(c::Candidate) = sum(dof, c.statistics)
StatsBase.r2(c::Candidate) = r2(c, :CoxSnell)

get_parameters(c::Candidate) = transform_parameter(c.parameterdist, c.parameters)
get_scales(c::Candidate) = transform_scales(c.observed, c.scales)


function Candidate(model, ps, rng, basis, dataset;
                   observed = ObservedModel(dataset.y),
                   parameterdist = ParameterDistributions(basis),
                   ptype = Float32)
    @unpack y, x = dataset

    T = eltype(dataset)

    # Create the initial state and path
    dataset_intervals = interval_eval(basis, dataset, get_interval(parameterdist))
    
    incoming_path = [PathState{ptype}(dataset_intervals[i], (), ((0, i),))
                     for i in 1:length(basis)]

    st = Lux.initialstates(rng, model)
    outgoing_path, st = sample(model, incoming_path, ps, st) 

    parameters = T.(get_init(parameterdist))
    scales = T.(get_init(observed))

    ŷ, _ = model(basis(dataset, transform_parameter(parameterdist, parameters)), ps, st)
    
    lls = logpdf(observed, y, ŷ, transform_scales(observed, scales))
    lls .+= logpdf(parameterdist, parameters)
    
    e = y .- ỹ

    rss = sum.(abs2, eachrow(e))
    dof_ = get_dof(outgoing_path)

    ȳ = mean(y, dims = 2)[:, 1]

    foreach(axes(y, 2)) do i
        ŷ[:, i] .= ȳ
    end

    null_ll = logpdf(observed, y, ŷ, transform_scales(observed, scales))

    stats = PathStatistics(
        rss, lls, null_ll, dof_, prod(size(y))
    )
    
    return Candidate{typeof(st)}(st, incoming_path, outgoing_path, observed,
                                             parameterdist,
                                             ll, null_ll, rss, dof_, prod(size(y)), scales,
                                             parameters,
                                             model, basis)
end

function update_values!(c::Candidate, ps, dataset)
    @unpack observed, st, scales, statistics, parameters, parameterdist, outgoing_path = c
    @unpack y = dataset
    
    ŷ = c(dataset, ps, st, parameters)
    
    dataloglikelihood = logpdf(observed, y, ŷ, transform_scales(observed, scales)) +
                          logpdf(parameterdist, parameters)
    rss = sum(abs2, y .- ŷ)
    dof = get_dof(outgoing_path)

    update_stats!(statistics, rss, dataloglikelihood, dof)
    return
end



@views function lossfunction(c::Candidate, p, st, ps::ComponentVector,
                             dataset::Dataset{T}) where {T}
    @unpack observed, parameterdist = c
    @unpack scales, parameters = ps
    @unpack y = dataset

    ll = logpdf(observed, y,
                c(dataset, p, st,transform_parameter(parameterdist, parameters)),
                transform_scales(observed, scales))
    ll += logpdf(parameterdist, parameters)
    -ll
end

function initial_values(c::Candidate)
    @unpack scales, parameters = c
    ComponentVector((; scales = scales, parameters = parameters))
end

function optimize_candidate!(c::Candidate, ps, dataset::Dataset{T}, optimizer,
                             options::Optim.Options) where {T}
    path, st = sample(c, ps)
    p_init = initial_values(c)

    if all(IntervalArithmetic.iscommon, map(get_interval, c.outgoing_path))
        if any(needs_optimization,(c.observed, c.parameterdist))
            loss(p) = lossfunction(c, ps, st, p, dataset)
            # We do not want any warnings here
            res = with_logger(NullLogger()) do 
                Optim.optimize(loss, p_init, optimizer, options)
            end

            if Optim.converged(res)
                c.outgoing_path .= path
                c.st = st
                c.parameters .= res.minimizer.parameters
                c.scales .= res.minimizer.scales
                update_values!(c, ps, dataset)
                return
            end
        else
            update_values!(c, ps, dataset)
        end
    end

    return
end

function check_intervals(paths::AbstractArray{<:AbstractPathState})::Bool
    @inbounds for path in paths
        check_intervals(path) || return false
    end
    return true
end

function sample(c::Candidate, ps, i = 0, max_sample = 10)
    @unpack incoming_path, st = c
    return sample(c.model.model, incoming_path, ps, st, i, max_sample)
end

function sample(model, incoming, ps, st, i = 0, max_sample = 10)
    outgoing, new_st = model(incoming, ps, st)
    if check_intervals(outgoing) || (i >= max_sample)
        return outgoing, new_st
    end
    return sample(model, incoming, ps, st, i + 1, max_sample)
end

get_nodes(c::Candidate) = ChainRulesCore.@ignore_derivatives get_nodes(c.outgoing_path)

