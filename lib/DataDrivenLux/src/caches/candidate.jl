mutable struct PathStatistics{T} <: StatsBase.StatisticalModel
    rss::T
    loglikelihood::T
    nullloglikelihood::T
    dof::Int
    nobs::Int
end

function update_stats!(
        stats::PathStatistics{T}, rss::T, ll::T, nullll::T, dof::Int) where {T}
    stats.dof = dof
    stats.loglikelihood = ll
    stats.nullloglikelihood = nullll
    stats.rss = rss
    return
end

StatsBase.rss(stats::PathStatistics) = getfield(stats, :rss)
StatsBase.nobs(stats::PathStatistics) = getfield(stats, :nobs)
StatsBase.loglikelihood(stats::PathStatistics) = getfield(stats, :loglikelihood)
StatsBase.nullloglikelihood(stats::PathStatistics) = getfield(stats, :nullloglikelihood)
StatsBase.dof(stats::PathStatistics) = getfield(stats, :dof)
StatsBase.r2(c::PathStatistics) = r2(c, :CoxSnell)

struct ComponentModel{B, M}
    basis::B
    model::M
end

function (c::ComponentModel)(dataset::Dataset{T}, ps, st::NamedTuple{fieldnames},
        p::AbstractVector{T}) where {T, fieldnames}
    return first(c.model(c.basis(dataset, p), ps, st))
end
function (c::ComponentModel)(ps, st::NamedTuple{fieldnames},
        paths::Vector{<:AbstractPathState}) where {fieldnames}
    return get_loglikelihood(c.model, ps, st, paths)
end

"""
$(TYPEDEF)

A container holding all the information for the current candidate solution
to the symbolic regression problem.

# Fields
$(FIELDS)
"""
struct Candidate{S <: NamedTuple} <: StatsBase.StatisticalModel
    "Random seed"
    rng::AbstractRNG
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

function (c::Candidate)(dataset::Dataset{T}, ps = c.ps, p = c.parameters) where {T}
    return c.model(dataset, ps, c.st, transform_parameter(c.parameterdist, p))
end
(c::Candidate)(ps = c.ps) = c.model(ps, c.st, c.outgoing_path)

Base.print(io::IO, c::Candidate) = print(io, "Candidate $(rss(c))")
Base.show(io::IO, c::Candidate) = print(io, c)
Base.summary(io::IO, c::Candidate) = print(io, c)

StatsBase.rss(c::Candidate) = rss(c.statistics)
StatsBase.nobs(c::Candidate) = nobs(c.statistics)
StatsBase.loglikelihood(c::Candidate) = loglikelihood(c.statistics)
StatsBase.nullloglikelihood(c::Candidate) = nullloglikelihood(c.statistics)
StatsBase.dof(c::Candidate) = dof(c.statistics)
StatsBase.r2(c::Candidate) = r2(c, :CoxSnell)

get_parameters(c::Candidate) = transform_parameter(c.parameterdist, c.parameters)
get_scales(c::Candidate) = transform_scales(c.observed, c.scales)

function Candidate(rng, model, basis, dataset; observed = ObservedModel(dataset.y),
        parameterdist = ParameterDistributions(basis), ptype = Float32)
    (; y, x) = dataset

    T = eltype(dataset)

    # Create the initial state and path
    dataset_intervals = interval_eval(basis, dataset, get_interval(parameterdist))

    incoming_path = [PathState{ptype}(dataset_intervals[i], (), ((0, i),))
                     for i in 1:length(basis)]

    ps, st = Lux.setup(rng, model)
    outgoing_path, st = sample(model, incoming_path, ps, st)
    ps = ComponentVector(ps)

    parameters = T.(get_init(parameterdist))
    scales = T.(get_init(observed))

    ŷ, _ = model(basis(dataset, transform_parameter(parameterdist, parameters)), ps, st)

    lls = logpdf(observed, y, ŷ, scales)
    lls += logpdf(parameterdist, parameters)

    rss = sum(abs2, y .- ŷ)
    dof_ = get_dof(outgoing_path)

    ȳ = vec(mean(y, dims = 2))

    null_ll = logpdf(observed, y, ȳ, scales) + logpdf(parameterdist, parameters)

    stats = PathStatistics(rss, lls, null_ll, dof_, prod(size(y)))

    return Candidate{typeof(st)}(
        Lux.replicate(rng), st, ComponentVector(ps), incoming_path, outgoing_path, stats,
        observed, parameterdist, scales, parameters, ComponentModel(basis, model))
end

function update_values!(c::Candidate, ps, dataset)
    (; observed, st, scales, statistics, parameters, parameterdist, outgoing_path) = c
    (; y) = dataset

    ŷ = c(dataset, ps, parameters)

    dataloglikelihood = logpdf(observed, y, ŷ, scales) + logpdf(parameterdist, parameters)
    rss = sum(abs2, y .- ŷ)
    dof = get_dof(outgoing_path)
    ȳ = vec(mean(y, dims = 2))
    nullloglikelihood = logpdf(observed, y, ȳ, scales) + logpdf(parameterdist, parameters)
    update_stats!(statistics, rss, dataloglikelihood, nullloglikelihood, dof)
    return
end

@views function Distributions.logpdf(
        c::Candidate, p::ComponentVector, dataset::Dataset{T}, ps = c.ps) where {T}
    (; observed, parameterdist) = c
    (; scales, parameters) = p
    (; y) = dataset

    ŷ = c(dataset, ps, parameters)
    return logpdf(c, p, y, ŷ)
end

function Distributions.logpdf(c::Candidate, p::AbstractVector, y::AbstractMatrix{T},
        ŷ::AbstractMatrix{T}) where {T}
    (; scales, parameters) = p
    (; observed, parameterdist) = c

    return logpdf(observed, y, ŷ, scales) + logpdf(parameterdist, parameters)
end

function initial_values(c::Candidate)
    (; scales, parameters) = c
    return ComponentVector((; scales = scales, parameters = parameters))
end

function optimize_candidate!(
        c::Candidate, dataset::Dataset{T}, ps = c.ps; optimizer = Optim.LBFGS(),
        options::Optim.Options = Optim.Options()) where {T}
    path, st = sample(c, ps)
    p_init = initial_values(c)

    if all(IntervalArithmetic.iscommon, map(get_interval, c.outgoing_path))
        if any(needs_optimization, (c.observed, c.parameterdist))
            loss(p) = -logpdf(c, p, dataset)
            # We do not want any warnings here
            res = with_logger(NullLogger()) do
                return Optim.optimize(loss, p_init, optimizer, options)
            end

            if Optim.converged(res)
                c.outgoing_path .= path
                @set! c.st = st
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
    (; incoming_path, st) = c
    return sample(c.model.model, incoming_path, ps, st, i, max_sample)
end

function sample(model, incoming, ps, st, i = 0, max_sample = 10)
    outgoing, new_st = model(incoming, ps, st)
    if check_intervals(outgoing) || (i >= max_sample)
        return outgoing, new_st
    end
    return sample(model, incoming, ps, st, i + 1, max_sample)
end

get_nodes(c::Candidate) = @ignore_derivatives get_nodes(c.outgoing_path)

function convert_to_basis(
        candidate::Candidate, ps = candidate.ps, options = DataDrivenCommonOptions())
    (; basis, model) = candidate.model
    (; eval_expresssion) = options
    p_best = get_parameters(candidate)

    p_new = map(enumerate(ModelingToolkit.parameters(basis))) do (i, ps)
        return DataDrivenDiffEq._set_default_val(Num(ps), p_best[i])
    end

    subs = Dict(a => b for (a, b) in zip(ModelingToolkit.parameters(basis), p_new))

    rhs = map(x -> Num(x.rhs), equations(basis))
    eqs, _ = model(rhs, ps, candidate.st)

    eqs = collect(map(eq -> ModelingToolkit.substitute(eq, subs), eqs))

    return Basis(eqs, states(basis), parameters = p_new, iv = get_iv(basis),
        controls = controls(basis), observed = observed(basis),
        implicits = implicit_variables(basis),
        name = gensym(:Basis), eval_expression = eval_expresssion)
end
