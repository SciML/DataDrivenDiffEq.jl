
"""
$(TYPEDEF)

A container holding all information for the current configuration of an 
expression graph / solution candidate.
"""
struct ConfigurationCache{S <: NamedTuple, TS <: Real, T <: Real} <: AbstractConfigurationCache
    "The current state represented by a `NamedTuple`"
    st::S
    "The loglikelihood of the configuration"
    loglikelihood::TS
    "The degrees of freedom of the configuration, e.g. number of active nodes in the graph"
    graph_dof::Int
    "The observed error model"
    observed::ObservedModel
    "The parameter distributions"
    pdist::ParameterDistributions
    "The data loglikelihood of the configuration"
    dataloglikelihood::T
    "The data nullloglikelihood of the configuration"
    nullloglikelihood::T
    "The residual sum of squares"
    rss::T
    "The number of observations"
    nobs::Int
    "The optimal scales"
    scales::AbstractVector
    "The optimal parameters"
    parameters::AbstractVector
end

function ConfigurationCache(x::LayeredDAG, ps, st, basis::Basis, dataset::Dataset, 
    observed::ObservedModel = ObservedModel(size(dataset.y,1)), parameter_dist = ParameterDistributions(basis))

    @unpack y = dataset

    st_ = update_state(x, ps, st)
    
    graph_ll = logpdf(x, ps, st_)
    
    graph_dof = dof(x, ps, st_)
    
    p_init = get_init(parameter_dist)
    scale_init = get_init(observed)
    
    Ŷ, _ = x(basis(dataset, transform_parameter(parameter_dist, p_init)), ps, st_)

    data_ll = logpdf(observed, y, Ŷ, transform_scales(observed, scale_init))

    rss = sum(abs2, y .- Ŷ)

    # Lazy for now
    y_mean = mean(y, dims = 2)[:, 1]
    for i in axes(y, 2)
        Ŷ[:, i] .= y_mean
    end
    null_ll = logpdf(observed, y, Ŷ, transform_scales(observed, scale_init))

    return ConfigurationCache{typeof(st_), typeof(graph_ll), typeof(rss)}(st_, graph_ll, graph_dof, observed, parameter_dist, data_ll, null_ll, rss,
                              prod(size(y)), scale_init, p_init)
end

function Base.print(io::IO, c::ConfigurationCache)
    print(io, loglikelihood(c), "   ", dof(c), "    ", rss(c), "    ", aicc(c))
end

get_data_loglikelihood(c::ConfigurationCache) = getfield(c, :dataloglikelihood)
get_configuration_loglikelihood(c::ConfigurationCache) = getfield(c, :loglikelihood)
get_configuration_dof(c::ConfigurationCache) = getfield(c, :graph_dof)

function get_scales(c::ConfigurationCache)
    @unpack observed, scales = c
    transform_scales(observed, scales)
end

function get_parameters(c::ConfigurationCache)
    @unpack pdist, parameters = c
    transform_parameter(pdist, parameters)
end

StatsBase.loglikelihood(c::ConfigurationCache) = get_data_loglikelihood(c)
StatsBase.dof(c::ConfigurationCache) = get_configuration_dof(c) 
StatsBase.nullloglikelihood(c::ConfigurationCache) = getfield(c, :nullloglikelihood)
StatsBase.rss(c::ConfigurationCache) = getfield(c, :rss)
StatsBase.nobs(c::ConfigurationCache) = getfield(c, :nobs)
StatsBase.r2(c::ConfigurationCache) = r2(c, :CoxSnell)

@views function configuration_loss(p::AbstractVector{U},
                                   c::ConfigurationCache,
                                   chain::LayeredDAG, basis::Basis, dataset::Dataset{T},
                                   ps::Union{NamedTuple, AbstractVector}) where {U, T}
    @unpack st, observed, pdist = c
    @unpack scales, parameters = p

    @unpack y = dataset

    # Predict
    ŷ = first(chain(basis(dataset, transform_parameter(pdist, parameters)), ps, st))
    # Return value
    -logpdf(observed, y, ŷ, transform_scales(observed, scales)) - logpdf(pdist, parameters)
end

# Create the initial parameters for optimization

function intial_parameter_vector(c::ConfigurationCache)
    @unpack scales, parameters = c
    ComponentVector((; scales = scales, parameters = parameters))
end

function optimize_configuration!(c::ConfigurationCache, d::LayeredDAG, ps::Union{NamedTuple, AbstractVector},
                                 dataset::Dataset{T}, basis::Basis, optimizer = Optim.BFGS(),
                                 options = Optim.Options()) where {T}
    @unpack st, pdist = c
    
    assert_intervals(d, ps, st, basis, dataset, get_interval(pdist)) || return c

    loss(p) = configuration_loss(p, c, d, basis, dataset, ps)

    p_init = intial_parameter_vector(c)

    res = Optim.optimize(loss, p_init, optimizer, options, autodiff = :forward)

    if Optim.converged(res)
        c = update_configuration!(c, res)
        c = evaluate_configuration!(c, d, ps, basis, dataset)
        return c
    end

    return c
end

function update_configuration!(c::ConfigurationCache, res)
    c = @set! c.scales = res.minimizer.scales
    c = @set! c.parameters = res.minimizer.parameters
end

function evaluate_configuration!(c::ConfigurationCache, d::LayeredDAG, ps, basis::Basis, dataset::Dataset{T}) where T
    @unpack observed, st = c
    @unpack y = dataset
    Ŷ = first(d(basis(dataset, get_parameters(c)) , ps, st))
    c = @set! c.dataloglikelihood = logpdf(observed, y, Ŷ, get_scales(c))
    c = @set! c.rss = sum(abs2, y .- Ŷ)
    c = @set! c.nobs = prod(size(y))
    c = @set! c.nullloglikelihood = logpdf(observed, y, Ŷ, get_scales(c))
    return c
end

function resample!(c::ConfigurationCache, d::LayeredDAG, ps, dataset::Dataset, basis::Basis, optimizer = Optim.BFGS(),
                   options = Optim.Options())
    # Sample new state
    c = @set! c.st = update_state(d, ps, c.st)
    # Update the parameters
    return optimize_configuration!(c, d, ps, dataset, basis , optimizer, options)
end
