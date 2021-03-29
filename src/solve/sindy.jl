## Used for solving explicit sindy problems

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

# Main
function DiffEqBase.solve(p::DataDrivenProblem{dType}, b::Basis, opt::Optimize.AbstractOptimizer;
    normalize::Bool = false, denoise::Bool = false,
    round::Bool = true, kwargs...) where {dType <: Number}
    # Check the validity
    @assert is_valid(p) "The problem seems to be ill-defined. Please check the problem definition."

    # Evaluate the basis
    θ = b(DataDrivenDiffEq.get_oop_args(p)...)

    # Normalize via p norm
    scales = ones(dType, size(θ, 1))

    normalize ? normalize_theta!(scales, θ) : nothing

    # Denoise via optimal shrinkage
    denoise ? optimal_shrinkage!(θ') : nothing

    # Init the coefficient matrix
    Ξ = DataDrivenDiffEq.Optimize.init(opt, θ', p.DX')
    # Solve
    Optimize.sparse_regression!(Ξ, θ', p.DX', opt; kwargs...)

    normalize ? rescale_xi!(Ξ, scales, round) : nothing
    return Ξ
    # Build solution Basis
    return build_solution(
        p, Ξ, opt, b
    )
end

function _isin(x::Num, y)
    vs = get_variables(y)
    any(isequal(x, yi) for yi in vs)
end

function _isin(x::Vector{Num}, y::Vector)
    [_isin(xi, yi) for xi in x, yi in y]
end

function _ind_matrix(x::Vector{Num}, y::Vector)
    isins = _isin(x, y)
    inds = ones(Bool, size(isins)) # We take all
    excludes = zeros(Bool, length(x))
    for i in 1:length(x)
        excludes .= true
        excludes[i] = false
        inds[i, :] .= inds[i, :] .* sum(eachrow(isins[excludes, :]))
    end
    return .~ inds
end


function DiffEqBase.solve(p::DataDrivenProblem{dType}, b::Basis,
    opt::Optimize.AbstractSubspaceOptimizer, implicits::Vector{Num};
    normalize::Bool = false, denoise::Bool = false,
    round::Bool = true, kwargs...) where {dType <: Number}
    # Check the validity
    @assert is_valid(p) "The problem seems to be ill-defined. Please check the problem definition."
    @assert length(implicits) == size(p.DX, 1) "Please provide enought implicit variables for the given problem."
    # Check for the variables
    @assert all(any.(eachrow(_isin(implicits, states(b)))))

    # Evaluate the basis
    θ = b(DataDrivenDiffEq.get_implicit_oop_args(p)...)

    # Normalize via p norm
    scales = ones(dType, size(θ, 1))

    normalize ? normalize_theta!(scales, θ) : nothing

    # Denoise via optimal shrinkage
    denoise ? optimal_shrinkage!(θ') : nothing

    # Init the coefficient matrix
    Ξ = DataDrivenDiffEq.Optimize.init(opt, θ', p.DX')
    # Find the implict variables in the equations and
    # eliminite duplictes
    inds = _ind_matrix(implicits, [eq.rhs for eq in equations(b)])

    # Solve for each implicit variable
    @views for i in 1:size(inds, 1)
        Optimize.sparse_regression!(Ξ[inds[i,:], i:i], θ[inds[i,:],:]', p.DX[i:i, :]', opt; kwargs...)
    end

    normalize ? rescale_xi!(Ξ, scales, round) : nothing
    return Ξ
    # Build solution Basis
    return build_implicit_solution(
        p, Ξ, opt, b, du
    )
end
