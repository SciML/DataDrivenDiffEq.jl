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
