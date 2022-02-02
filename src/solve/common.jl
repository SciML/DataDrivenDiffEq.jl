"""
$(TYPEDEF)

Common options for all methods provided via `DataDrivenDiffEq`. 
"""
mutable struct DataDrivenCommonOptions{T}
    """Maximum iterations"""
    maxiter::Int
    """Absolute tolerance"""
    abstol::T 
    """Relative tolerance"""
    reltol::T
    
    """Show a progress"""
    progress::Bool
    """Display log - Not implemented right now"""
    verbose::Bool
    
    """Denoise the data using singular value decomposition"""
    denoise::Bool
    """Normalize the data"""
    normalize::Bool
    """Sample options, see `DataSampler`"""
    sampler::AbstractSampler

    """Mapping from the candidate solution of a problem to features used for pareto analysis"""
    f::Function
    """Scalarization of the features for a candidate solution"""
    g::Function
    """Additional kwargs"""
    kwargs::Base.Pairs
end

DataDrivenCommonOptions(opt::AbstractOptimizer{T}, args...; 
    maxiter = 100, abstol = sqrt(eps()), reltol = sqrt(eps()),
    progress = false, verbose = false, 
    denoise = false, normalize = false, 
    sampler = DataSampler(),
    f = F(opt), g = G(opt), kwargs...) where T = begin
   DataDrivenCommonOptions{eltype(T)}(
       maxiter, abstol, reltol, 
       progress, verbose, denoise, normalize, 
       sampler, f, g, kwargs
   ) 
end

## Normalization etc
function normalize_theta!(scales::AbstractVector, theta::AbstractMatrix)
    map(1:length(scales)) do i
        scales[i] = norm(theta[i,:], 2)
        theta[i, :] .= theta[i,:]./scales[i]
    end
    return
end

function rescale_xi!(xi::AbstractMatrix, scales::AbstractVector, round_::Bool)
    digs = 10
    @inbounds for i in 1:length(scales), j in 1:size(xi, 2)
        iszero(xi[i,j]) ? continue : nothing
        round_ && (xi[i,j] % 1) != zero(xi[i,j]) ? digs = round(Int64,-log10(abs(xi[i,j]) % 1))+1 : nothing
        xi[i,j] = xi[i,j] / scales[i]
        round_ ? xi[i,j] = round(xi[i,j], digits = digs) : nothing
    end
    return
end

#function DiffEqBase.init(prob::AbstractDataDrivenProblem{N,C,P}, basis::AbstractBasis, opt::AbstractOptimizer, options::DataDrivenCommonOptions, 
#    args...; kwargs...) where {N,C,P}
#    
#    @unpack normalize, denoise, sampler = options
#
#    dx = zeros(N, length(basis), length(prob))
#    
#    @views if isa(opt, AbstractSubspaceOptimizer) 
#        basis(dx, get_implicit_oop_args(prob)...)
#    else
#        basis(dx, prob)
#    end
#    
#    scales = ones(N, size(dx, 1))
#    
#    normalize ? normalize_theta!(scales, dx) : nothing
#
#    denoise ? optimal_shrinkage!(dx') : nothing
#
#    train, test = sampler(prob)
#
#    Y = get_target(prob)
#
#    n_y = size(Y, 1)
#
#
#    Ξ = zeros(N, length(train), length(basis) , n_y)
#    
#    return dx, Y, Ξ, train, test, scales
#end


