"""
$(TYPEDEF)

Common options for all methods provided via `DataDrivenDiffEq`. 
"""
@with_kw mutable struct DataDrivenCommonOptions{T}
    """Maximum iterations"""
    maxiter::Int = 100
    """Absolute tolerance"""
    abstol::T = sqrt(eps(T))
    """Relative tolerance"""
    reltol::T = sqrt(eps(T))
    
    """Show a progress"""
    progress::Bool = false
    """Display log - Not implemented right now"""
    verbose::Bool = false
    
    """Denoise the data using singular value decomposition"""
    denoise::Bool = false
    """Normalize the data"""
    normalize::Bool = false
    """Sample options, see `DataSampler`"""
    sampler::AbstractSampler = DataSampler()

    """Mapping from the candidate solution of a problem to features used for pareto analysis"""
    f::Function = (x, A, y, λ = zero(T)) -> [norm(x, 0); norm(y .- A*x, 2)]
    """Scalarization of the features for a candidate solution"""
    g::Function = f->f[1] < 1 ? Inf : norm(f, 2)
end

set_from_kwargs!(d::DataDrivenCommonOptions, kwargs...) = begin
    _keys = fieldnames(DataDrivenCommonOptions)
    for (f, v) in kwargs
        f ∈ _keys ? setfield!(d, f, v) : nothing
    end
    return 
end


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

mutable struct SparseLinearProblem{A,T,V,S,X,Y,C} 
    Θ::A
    trains::T
    val::V
    tests::S
    Ξ::X
    DX::Y
    scales::C
end

struct SparseLinearSolution{X, L, S, E, F, O}
    Ξ::X
    λ::L
    sets::S
    error::E
    folds::F
    opt::O
end

function SparseLinearSolution(p::SparseLinearProblem, errors, folds, λ, opt)
    @unpack Ξ, trains, tests = p
    return SparseLinearSolution(Ξ, λ, (trains, tests), errors, folds, opt)
end

# Selection
select_by(x, y::AbstractMatrix) = y 
select_by(x, sol) = select_by(Val(x), sol)

select_by(::Val, sol::SparseLinearSolution) = begin
    @unpack Ξ, error, λ = sol
    i = argmin(error)
    return Ξ[i,:,:], error[i], λ[i,:]
end

select_by(::Val{:kfold}, sol::SparseLinearSolution) = begin
    @unpack Ξ, folds, error, λ = sol
    size(Ξ, 1) <= 1 && return select_by(1, sol)
    i = argmin(mean(folds, dims = 1)[1,:])
    return Ξ[i,:,:], error[i], λ[i,:]
end

select_by(::Val{:stat}, sol::SparseLinearSolution) = begin
    @unpack Ξ, folds, error, λ = sol
    size(Ξ, 1) <= 1 && return select_by(1, sol)
    best = argmin(error)
    ξ = mean(Ξ, dims = 1)[1,:,:]
    s = std(Ξ, dims = 1)[1,:,:]
    return measurement.(ξ, s), error[best], λ[best,:]
end

function DiffEqBase.init(prob::AbstractDataDrivenProblem{N,C,P}, basis::AbstractBasis, opt::AbstractOptimizer, options::DataDrivenCommonOptions, 
    args...; kwargs...) where {N,C,P}
    
    @unpack normalize, denoise, sampler = options

    dx = zeros(N, length(basis), length(prob))
    
    @views basis(dx, prob)
    
    scales = ones(N, size(dx, 1))
    
    normalize ? normalize_theta!(scales, dx) : nothing

    denoise ? optimal_shrinkage!(dx') : nothing

    train, test = sampler(prob)

    Y = get_target(prob)

    n_y = size(Y, 1)

    Ξ = zeros(N, length(train), length(basis), n_y)

    return SparseLinearProblem{typeof(dx), typeof(train), typeof(nothing), typeof(test), typeof(Ξ), typeof(Y), typeof(scales)}(
        dx, train, nothing, test, Ξ, Y, scales
    )
end


function DiffEqBase.solve(p::AbstractDataDrivenProblem{T, C, P}, basis, opt::AbstractOptimizer, opts = DataDrivenCommonOptions{T}(), args...;
    eval_expression = false,  kwargs...) where {T,C,P}
    
    set_from_kwargs!(opts, kwargs...)

    prob = init(p, basis, opt, opts)
    
    @unpack Θ, DX, Ξ, trains, tests, scales = prob 
    @unpack maxiter, abstol, reltol, verbose, progress,f,g= opts
    
    testerror = zeros(T, size(Ξ, 1))
    trainerror = zeros(T, size(Ξ, 1), size(Ξ,1))
    λs = zeros(T, size(Ξ, 1), size(DX, 1))
    
    fg = (x...)->g(f(x...))
    
    Aₜ = Θ[:, tests]'
    Yₜ = DX[:, tests]'

    @views for (i,t) in enumerate(trains)

        A = Θ[:,t]'
        Y = DX[:, t]'
        X = Ξ[i, :, :] 

        λs[i,:] .= sparse_regression!(X, A, Y, opt, 
            maxiter = maxiter, abstol = abstol, f = f, g = g, progress = progress
        )

        for (j, tt) in enumerate(trains)
            trainerror[i,j] = fg(X, Θ[:,tt]', DX[:, tt]')
        end

        testerror[i] = fg(X, Aₜ, Yₜ)

        rescale_xi!(X, scales, true)
    end

    # Rescale
    sol = SparseLinearSolution(prob, testerror, trainerror, λs, opt)

    return DataDrivenSolution(p, sol, basis, opt; eval_expression = eval_expression, kwargs...)
end

