"""
$(TYPEDEF)

Common options for all methods provided via `DataDrivenDiffEq`. 

# Fields
$(FIELDS)
    
"""
mutable struct DataDrivenCommonOptions{T,K}
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
    """Significant digits for the parameters - used for rounding. Default = 10"""
    digits::Int 

    """Additional kwargs"""
    kwargs::K
end

DataDrivenCommonOptions(opt::AbstractKoopmanAlgorithm, ::Type{T} = Float64, args...; 
    maxiter = 100, abstol = sqrt(eps()), reltol = sqrt(eps()),
    progress = false, verbose = false, 
    denoise = false, normalize = false, 
    sampler = DataSampler(),
    f = F(opt), g = G(opt), digits = 10, kwargs...) where T = begin
   DataDrivenCommonOptions{eltype(T), typeof(kwargs)}(
       maxiter, abstol, reltol, 
       progress, verbose, denoise, normalize, 
       sampler, f, g, digits, kwargs
   ) 
end

DataDrivenCommonOptions(opt::AbstractOptimizer{T}, args...; 
    maxiter = 100, abstol = sqrt(eps()), reltol = sqrt(eps()),
    progress = false, verbose = false, 
    denoise = false, normalize = false, 
    sampler = DataSampler(),
    f = F(opt), g = G(opt), 
    digits = 10,
    kwargs...) where T = begin
   DataDrivenCommonOptions{eltype(T), typeof(kwargs)}(
       maxiter, abstol, reltol, 
       progress, verbose, denoise, normalize, 
       sampler, f, g, digits, kwargs
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
        xi[i,j] = xi[i,j] / scales[i]
        round_ && (xi[i,j] % 1) != zero(xi[i,j]) ? digs = round(Int64,-log10(abs(xi[i,j]) % 1))+1 : nothing
        round_ ? xi[i,j] = round(xi[i,j], digits = digs) : nothing
    end
    return
end

function candidate_matrix(b::Basis, n_o::Int)
    eqs = map(x->x.rhs, equations(b))
    xs = states(b)
    ys = implicit_variables(b)

    isempty(ys) && return ones(Bool, n_o, length(eqs))
    c = zeros(Bool, length(ys), length(eqs))

    for i in 1:length(ys), j in 1:length(eqs)
        # Either we have a dependency on this variable
        c[i,j] = is_dependent(Num(eqs[j]), Num(ys[i]))
        # Return 
        c[i,j] && continue
        # Or to no other implicit variable
        c[i,j] = all(map(xi->is_not_dependent(Num(eqs[j]), Num(xi)), ys))
    end

    return c
end