"""
$(TYPEDEF)

A container holding all the information for the current candidate solution
to the symbolic regression problem.

# Fields
$(FIELDS)
"""
mutable struct Candidate{S <: NamedTuple, T <: Real} <: StatsBase.StatisticalModel
    "The current state"
    st::S
    "Incoming paths"
    incoming_path::Vector{AbstractPathState}
    "Outgoing path"
    outgoing_path::Vector{AbstractPathState}
    "The observed model"
    observed::ObservedModel
    "The parameter distribution"
    parameterdist::ParameterDistributions
    "Dataloglikelihood"
    dataloglikelihood::T
    "Nullloglikelihood"
    nullloglikelihood::T
    "Residual sum of squares"
    rss::T
    "Degrees of freedom"
    dof::Int
    "Number of observations"
    nobs::Int
    "The optimal scales"
    scales::AbstractVector
    "The optimal parameters" 
    parameters::AbstractVector
    "The graph model"
    model::Lux.AbstractExplicitContainerLayer
    "The basis"
    basis::Basis
end

Base.print(io::IO, c::Candidate) = print(io, "Candidate $(rss(c))")
Base.show(io::IO, c::Candidate) = print(io, c)
Base.summary(io::IO, c::Candidate) = print(io, c)

function Candidate(model, ps, st_, basis, dataset; 
        observed = ObservedModel(size(dataset.y,1)), 
        parameterdist = ParameterDistributions(basis),
        ptype = Float32,
    )

    @unpack y, x = dataset

    T = eltype(dataset)
    # Create the initial state and path
    incoming_path = [PathState{ptype}(zero(ptype), (), ((0,i),)) for i in 1:length(basis)]
    outgoing_path, st = model(incoming_path, ps, st_)

    parameters = T.(get_init(parameterdist))
    scales = T.(get_init(observed))

    ŷ, _ = model(basis(dataset, transform_parameter(parameterdist, parameters)), ps, st)

    ll = logpdf(observed, y, ŷ, transform_scales(observed, scales))
    ll += logpdf(parameterdist, parameters)

    rss = sum(abs2, y .- ŷ)
    dof_ = get_dof(outgoing_path)

    ȳ = mean(y, dims = 2)[:,1]
    
    foreach(axes(y,2)) do i
        ŷ[:,i] .= ȳ
    end

    null_ll = logpdf(observed, y, ŷ, transform_scales(observed, scales))
    return Candidate{typeof(st), typeof(ll)}(
        st, incoming_path, outgoing_path, observed, parameterdist, 
        ll, null_ll, rss, dof_, prod(size(y)), scales, parameters, 
        model, basis
    )
end

(c::Candidate)(dataset::Dataset, ps, st::NamedTuple, p::AbstractVector) = first(c.model(c.basis(dataset, p), ps, st))

function update_values!(c::Candidate, ps, dataset)
    @unpack observed, st , scales, parameters, parameterdist, outgoing_path = c
    @unpack y = dataset
    ŷ = c(dataset, ps, st, transform_parameter(parameterdist, parameters))
    c.dataloglikelihood = logpdf(observed, y, ŷ, transform_scales(observed, scales)) + logpdf(parameterdist, parameters)
    c.rss = sum(abs2, y .- ŷ)
    c.nobs = prod(size(y))
    c.dof = get_dof(outgoing_path)
    ȳ = mean(y, dims = 2)[:,1]
    foreach(axes(y,2)) do i
        ŷ[:,i] .= ȳ
    end
    c.nullloglikelihood = logpdf(observed, y, ŷ, transform_scales(observed, scales))
    return 
end

StatsBase.loglikelihood(c::Candidate) = getfield(c, :dataloglikelihood)
StatsBase.dof(c::Candidate) = getfield(c, :dof)
StatsBase.nullloglikelihood(c::Candidate) = getfield(c, :nullloglikelihood)
StatsBase.rss(c::Candidate) = getfield(c, :rss)
StatsBase.nobs(c::Candidate) = getfield(c, :nobs)
StatsBase.r2(c::Candidate) = r2(c, :CoxSnell)

get_parameters(c::Candidate) = transform_parameter(c.parameterdist, c.parameters)

@views function lossfunction(c::Candidate, p, st, ps::ComponentVector, dataset::Dataset{T}) where T
    @unpack observed, parameterdist = c
    @unpack scales, parameters = ps
    @unpack y = dataset

    ll = logpdf(observed, y, c(dataset, p, st, 
        transform_parameter(parameterdist, parameters)), 
        transform_scales(observed, scales))
    ll += logpdf(parameterdist, parameters)
    -ll
end

function initial_values(c::Candidate) 
    @unpack scales, parameters = c
    ComponentVector((; scales = scales, parameters = parameters))
end

function optimize_candidate!(c::Candidate, ps, dataset::Dataset{T}, optimizer, options::Optim.Options) where T    
    
    path, st = sample!(c, ps)

    p_init = initial_values(c)

    loss(p) = lossfunction(c, ps, st, p, dataset)

    res = Optim.optimize(loss, p_init, optimizer, options)

    if Optim.converged(res)
        c.outgoing_path .= path
        c.st = st
        c.parameters .= res.minimizer.parameters
        c.scales .= res.minimizer.scales
        return 
    end

    return 
end

function sample!(c::Candidate, ps)
    @unpack incoming_path, st = c
    outgoing, new_st = c.model(incoming_path, ps, st)
end

get_nodes(c::Candidate) = ChainRulesCore.@ignore_derivatives get_nodes(c.outgoing_path)

function get_loglikelihood(c::Candidate, ps)
    DataDrivenLux.get_loglikelihood(c.model, ps, c.st, get_nodes(c))
end

# We simply assume here the loglikelihood is meant
(c::Candidate)(ps) = get_loglikelihood(c, ps)
