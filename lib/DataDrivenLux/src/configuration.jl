
"""
using Base: dataids
$(TYPEDEF)

A container holding all information for the current configuration of an 
expression graph / solution candidate.
"""
struct ConfigurationCache{S <: NamedTuple, L <: Number, D <: Number, SC <: AbstractVector, TS,  DP, P, TP} <: AbstractConfigurationCache
    "The current state represented by a `NamedTuple`"
    st::S
    "The loglikelihood of the configuration"
    loglikelihood::L
    "The degrees of freedom of the configuration, e.g. number of active nodes in the graph"
    graph_dof::Int
    "The observed error model"
    errormodel::ObservedError
    "The data loglikelihood of the configuration"
    dataloglikelihood::D
    "The data nullloglikelihood of the configuration"
    nullloglikelihood::D
    "The residual sum of squares"
    rss::D
    "The number of observations"
    nobs::Int
    "The scales of the error model"
    scales::SC
    "The transformation of the scales"
    transform_scales::TS
    "The vector of optimal parameters"
    p::P
    "The distributions of the optimal parameters"
    pdist::DP
    "The transformation of the optimal parameters to its domain"
    transform_parameters::TP
end

function ConfigurationCache(x::LayeredDAG, ps, st, basis::Basis, dataset::Dataset;
                            errormodel = ObservedError(size(dataset.y,1)),
                            scales = zeros(eltype(dataset.y), size(dataset.y, 1)),
                            transform_scales = as(Array, as(Real, 1e-5, TransformVariables.∞), size(scales)),
                            pdist = nothing, transform_parameters = nothing,
                            kwargs...)

    @assert isnothing(pdist) ? isnothing(transform_parameters) : true 
    @unpack y = dataset

    st_ = update_state(x, ps, st)
    graph_ll = logpdf(x, ps, st_)
    graph_dof = dof(x, ps, st_)
    
    p_init = isnothing(pdist) ? [] : map(inverse, transform_parameters, map(rand, pdist))
    
    Ŷ, _ = x(basis(dataset, p_init), ps, st_)

    scales_ = transform(transform_scales, scales)

    data_ll = logpdf(errormodel, y, Ŷ, scales_)
    rss = sum(abs2, y .- Ŷ)
    # Lazy for now
    y_mean = mean(y, dims = 2)[:, 1]
    for i in axes(y, 2)
        Ŷ[:, i] .= y_mean
    end
    null_ll = logpdf(errormodel, y, Ŷ, scales_)

    return ConfigurationCache(st_, graph_ll, graph_dof, errormodel, data_ll, null_ll, rss,
                              prod(size(y)), scales, transform_scales, p_init, pdist,
                              transform_parameters)
end

function Base.print(io::IO, c::ConfigurationCache)
    print(io, loglikelihood(c), "   ", dof(c), "    ", rss(c), "    ", aicc(c))
end

get_data_loglikelihood(c::ConfigurationCache) = getfield(c, :dataloglikelihood)
get_configuration_loglikelihood(c::ConfigurationCache) = getfield(c, :loglikelihood)
get_configuration_dof(c::ConfigurationCache) = getfield(c, :graph_dof)

function get_scales(c::ConfigurationCache)
    transform(getfield(c, :transform_scales), getfield(c, :scales))
end

function get_parameters(c::ConfigurationCache)
    @unpack p, pdist, transform_parameters = c
    isnothing(pdist) && return p
    map(transform, transform_parameters, p)
end

StatsBase.loglikelihood(c::ConfigurationCache) = get_data_loglikelihood(c)
StatsBase.dof(c::ConfigurationCache) = get_configuration_dof(c)
StatsBase.nullloglikelihood(c::ConfigurationCache) = getfield(c, :nullloglikelihood)
StatsBase.rss(c::ConfigurationCache) = getfield(c, :rss)
StatsBase.nobs(c::ConfigurationCache) = getfield(c, :nobs)
StatsBase.r2(c::ConfigurationCache) = r2(c, :CoxSnell)

@views function configuration_loss(p::AbstractVector{U},
                                   c::ConfigurationCache{<:Any, <:Any, <:Any, <:Any, <:Any,
                                                         Nothing},
                                   chain::LayeredDAG, basis::Basis, dataset::Dataset{T},
                                   ps::Union{NamedTuple, AbstractVector}) where {U, T}
    @unpack transform_scales, st, errormodel = c
    @unpack scales = p

    @unpack y = dataset

    Ŷ, _ = chain(basis(dataset, []), ps, st)

    -logpdf(errormodel, y,
            Ŷ,
            transform(transform_scales, scales))
end

@views function configuration_loss(p::AbstractVector{U},
    c::ConfigurationCache,
    chain::LayeredDAG, basis::Basis, dataset::Dataset{T},
    ps::Union{NamedTuple, AbstractVector}) where {U, T}
    @unpack transform_scales, st, errormodel, transform_parameters, pdist = c
    @unpack scales, parameters = p
    @unpack y = dataset

    p_transformed = map(transform, transform_parameters, parameters)

    Ŷ, _ = chain(basis(dataset, p_transformed) , ps, st)

    -logpdf(errormodel, y,
        Ŷ,
        transform(transform_scales, scales)
        ) - sum(map(logpdf, pdist, p_transformed))
end

# Create the initial parameters for optimization
function _init_ps(c::ConfigurationCache{<:Any, <:Any, <:Any, <:Any, <:Any, Nothing})
    @unpack scales = c
    ComponentVector((; scales = scales))
end

function _init_ps(c::ConfigurationCache)
    @unpack scales, p = c
    ComponentVector((; scales = scales, parameters = p))
end

function optimize_configuration!(c::ConfigurationCache, d::LayeredDAG, ps,
                                 dataset::Dataset{T}, basis::Basis, optimizer = Optim.BFGS(),
                                 options = Optim.Options()) where {T}
    @unpack st, p = c
    
    assert_intervals(d, ps, st, basis, dataset, get_parameters(c)) || return c

    loss(p) = configuration_loss(p, c, d, basis, dataset, ps)

    p_init = _init_ps(c)

    res = Optim.optimize(loss, p_init, optimizer, options, autodiff = :forward)

    if Optim.converged(res)
        c = update_configuration!(c, res)
        c = evaluate_configuration!(c, d, ps, basis, dataset)
        return c
    end

    return c
end

function update_configuration!(c::ConfigurationCache{<:Any, <:Any, <:Any, <:Any, <:Any,
                                                     Nothing}, res)
    c = @set! c.scales = res.minimizer.scales
end

function update_configuration!(c::ConfigurationCache, res)
    c = @set! c.scales = res.minimizer.scales
    c = @set! c.p = res.minimizer.parameters
end

function evaluate_configuration!(c::ConfigurationCache, d::LayeredDAG, ps, basis::Basis, dataset::Dataset{T}) where T
    @unpack errormodel, st = c
    @unpack y = dataset
    Ŷ = first(d(basis(dataset, get_parameters(c)) , ps, st))
    c = @set! c.dataloglikelihood = logpdf(errormodel, y, Ŷ, get_scales(c))
    c = @set! c.rss = sum(abs2, y .- Ŷ)
    c = @set! c.nobs = prod(size(y))
    c = @set! c.nullloglikelihood = logpdf(errormodel, y, Ŷ, get_scales(c))
    return c
end

function resample!(c::ConfigurationCache, d::LayeredDAG, ps, dataset::Dataset, basis::Basis, optimizer = Optim.BFGS(),
                   options = Optim.Options())
    # Sample new state
    c = @set! c.st = update_state(d, ps, c.st)
    # Update the parameters
    return optimize_configuration!(c, d, ps, dataset, basis , optimizer, options)
end
