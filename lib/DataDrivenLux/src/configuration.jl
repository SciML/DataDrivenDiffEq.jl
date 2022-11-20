
"""
$(TYPEDEF)

A container holding all information for the current configuration of an 
expression graph / solution candidate.
"""
struct ConfigurationCache{S, L, D, SC, TS, P, DP, TP} <: AbstractConfigurationCache
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

function ConfigurationCache(x::LayeredDAG, ps, st, X::AbstractArray, Y::AbstractArray; 
    errormodel = ObservedError(size(Y, 1)), 
    scales = zeros(eltype(Y), size(Y, 1)), 
    transform_scales = as(Array, as(Real, 1e-5, ∞), size(scales)), 
    kwargs...)

    # Use this later with a basis
    #parameters = eltype(Y)[], transform_parameters = as(typeof(parameters),asℝ, size(parameters)))

    st_ = update_state(x, ps, st)
    graph_ll = logpdf(x, ps, st_)
    graph_dof = dof(x, ps, st_)
    Ŷ, _ = x(X, ps, st_)
    
    scales_ = transform(transform_scales, scales)

    data_ll = logpdf(errormodel, Y, Ŷ ,scales_)
    rss = sum(abs2, Y .- Ŷ)
    # Lazy for now
    y_mean = mean(Y, dims =2)[:,1]
    for i in axes(Y, 2)
        Ŷ[:,i] .= y_mean
    end
    null_ll = logpdf(errormodel, Y, Ŷ, scales_)

    return ConfigurationCache(
        st_, graph_ll, graph_dof, errormodel, data_ll, null_ll, rss, prod(size(Y)), scales, transform_scales, nothing, nothing, nothing
    )
end



get_data_loglikelihood(c::ConfigurationCache) = getfield(c, :dataloglikelihood)
get_configuration_loglikelihood(c::ConfigurationCache) = getfield(c, :loglikelihood)
get_configuration_dof(c::ConfigurationCache) = getfield(c, :graph_dof)
get_scales(c::ConfigurationCache) = transform(getfield(c, :transform_scales), getfield(c, :scales))

StatsBase.loglikelihood(c::ConfigurationCache) = get_data_loglikelihood(c)
StatsBase.dof(c::ConfigurationCache) = get_configuration_dof(c)
StatsBase.nullloglikelihood(c::ConfigurationCache) = getfield(c, :nullloglikelihood)
StatsBase.rss(c::ConfigurationCache) = getfield(c, :rss)
StatsBase.nobs(c::ConfigurationCache) = getfield(c, :nobs)
StatsBase.r2(c::ConfigurationCache) = r2(c, :CoxSnell)

function _generate_loss(c::ConfigurationCache{<:Any, <:Any, <:Any, <:Any, <:Any, Nothing}, chain::LayeredDAG, ps, X, Y)
    @unpack transform_scales, st, errormodel = c
    loss = let tf = transform_scales, errormodel_ = errormodel, st_ = st, X = X, Y = Y, ps = ps
        (p)-> begin 
            σ = transform(tf, p.scales)
            Ŷ, _ = chain(X, ps, st_)
            -logpdf(errormodel_, Y, Ŷ, σ)
        end
    end
end

# Create the initial parameters for optimization
function _init_ps(c::ConfigurationCache{<:Any, <:Any, <:Any, <:Any, <:Any, Nothing})
    @unpack scales = c
    ComponentVector((; scales = scales))
end

function optimize_configuration!(c::ConfigurationCache, d::LayeredDAG, ps, X::AbstractArray, Y::AbstractArray, optimizer = Optim.BFGS(), options = Optim.Options())
    loss = _generate_loss(c, d, ps, X, Y)
    p_init = _init_ps(c)
    res = Optim.optimize(loss, p_init, optimizer, options)
    if Optim.converged(res)
        c = update_configuration!(c, res)
        c = evaluate_configuration!(c, d, ps, X, Y)
        return c
    end
    return c
end

function update_configuration!(c::ConfigurationCache{<:Any, <:Any, <:Any, <:Any, <:Any, Nothing}, res)
    c = @set! c.scales = res.minimizer.scales
end

function evaluate_configuration!(c::ConfigurationCache, d::LayeredDAG, ps, X::AbstractArray, Y::AbstractArray)
    @unpack errormodel, st = c
    Ŷ = first(d(X, ps, st))
    y_mean = mean(Y, dims=2)[:,1]
    c = @set! c.dataloglikelihood = logpdf(errormodel, Y, Ŷ, get_scales(c))
    c = @set! c.rss = sum(abs2, Y .- Ŷ)
    c = @set! c.nobs = prod(size(Y))
    foreach(axes(Y, 2)) do i 
        Ŷ[:,i] .= y_mean
    end
    c = @set! c.nullloglikelihood = logpdf(errormodel, Y, Ŷ, get_scales(c))
    return c 
end