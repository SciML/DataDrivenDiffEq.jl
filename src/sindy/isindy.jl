# TODO I think here is some potential for faster computation
# However, up to 25 states, the algorithm works fine and fast (main knobs are rtol and maxiter)
# This is the specialized version assuming a mass matrix form / linear in dx
# M(x, p)*dx = f(x, p)
# Where M is diagonal!

# TODO preallocation

"""
    ISInDy(X, Y, Ψ; weights, f_target, maxiter, rtol, p, t, opt)

Performs an implicit sparse identification of nonlinear dynamics given the data matrices `X` and `Y` via the `AbstractBasis` `basis.`
Keyworded arguments include the parameter (values) of the basis `p` and the timepoints `t`, which are passed in optionally.
`opt` is an `AbstractSubspaceOptimizer` useable for sparse regression within the nullspace, `maxiter` the maximum iterations to perform, and `convergence_error` the
bound which causes the optimizer to stop.

The best vectors of the sparse nullspace are selected via multi-objective optimization.
The best candidate is determined via the `AbstractScalarizationMethod` given in `alg`. The evaluation has two fields, the sparsity of the coefficients and the L2-Norm error at index 1 and 2, respectively.

`rtol` gets directly passed into the computation of the nullspace.

Returns a `SInDyResult`.
"""
function ISInDy(X::AbstractArray, Ẋ::AbstractArray, Ψ::Basis; alg::Optimize.AbstractScalarizationMethod = WeightedSum(), maxiter::Int64 = 10, rtol::Float64 = 0.99, p::AbstractArray = [], t::AbstractVector = [], opt::T = ADM()) where T <: DataDrivenDiffEq.Optimize.AbstractSubspaceOptimizer
    @assert size(X)[end] == size(Ẋ)[end]

    # Compute the library and the corresponding nullspace
    θ = Ψ(X, p, t)

    # Init for sweep over the differential variables
    Ξ = zeros(eltype(θ), length(Ψ)*2, size(Ẋ, 1))

    opt_front = ParetoFront(1, scalarization = alg)
    tmp_front = ParetoFront(1, scalarization = alg)


    @inbounds for i in 1:size(Ẋ, 1)
        dθ = hcat(map((dxi, ti)->dxi.*ti, Ẋ[i, :], eachcol(θ))...)
        Θ = vcat(dθ, θ)
        N = nullspace(Θ', rtol = rtol)
        Q = deepcopy(N) # Deepcopy for inplace

        # Find sparse vectors in nullspace
        # Calls effectively the ADM algorithm with varying initial conditions
        DataDrivenDiffEq.fit!(Q, N', opt, maxiter = maxiter)

        # Compute pareto front
        @inbounds for (j, qi) in enumerate(eachcol(Q))
            if j < 2
                set_candidate!(opt_front, 1, [norm(qi, 0); norm(Θ'*qi, 2)], qi, maxiter, get_threshold(opt))
                set_candidate!(tmp_front, 1, [norm(qi, 0); norm(Θ'*qi, 2)], qi, maxiter, get_threshold(opt))
            else
                set_candidate!(tmp_front, 1, [norm(qi, 0); norm(Θ'*qi, 2)], qi, maxiter, get_threshold(opt))
                conditional_add!(opt_front, tmp_front)
            end
        end
        Ξ[:, i] .= Optimize.parameter(opt_front[1])
        Ξ[:, i] .= Ξ[:, i] ./ maximum(abs.(Ξ[:, i]))
    end

    return ImplicitSparseIdentificationResult(Ξ, Ψ, maxiter, opt, true, Ẋ, X, p = p, t = t)
end


function ImplicitSparseIdentificationResult(coeff::AbstractArray, equations::Basis, iters::Int64, opt::T, convergence::Bool, Y::AbstractVecOrMat, X::AbstractVecOrMat; p::AbstractArray = [], t::AbstractVector = []) where T <: Union{Optimize.AbstractOptimizer, Optimize.AbstractSubspaceOptimizer}

    sparsities = Int64.(norm.(eachcol(coeff), 0))

    b_, p_ = derive_implicit_parameterized_eqs(coeff, equations)
    ps = [p; p_]

    Ŷ = b_(X, ps, t)
    training_error = norm.(eachrow(Y-Ŷ), 2)
    aicc = similar(training_error)

    for i in 1:length(aicc)
        aicc[i] = AICC(sum(sparsities[i]), view(Ŷ, i, :) , view(Y, i, :))
    end
    return SparseIdentificationResult(coeff, [p...;p_...], b_ , opt, iters, convergence,  training_error, aicc,  sparsities)
end



function derive_implicit_parameterized_eqs(Ξ::AbstractArray{T, 2}, b::Basis) where T <: Real

    sparsity = Int64(norm(Ξ, 0))

    @parameters p[1:sparsity]
    p_ = zeros(eltype(Ξ), sparsity)
    cnt = 1

    b_ = Basis(Operation[], variables(b), parameters = [parameters(b)...; p...])

    for i=1:size(Ξ, 2)
        eq_d = nothing
        eq_n = nothing
        # Denominator
        for j = 1:length(b)
            if !iszero(Ξ[j,i])
                if eq_d === nothing
                    eq_d = p[cnt]*b[j]
                else
                    eq_d += p[cnt]*b[j]
                end
                p_[cnt] = Ξ[j,i]
                cnt += 1
            end
        end
        # Numerator
        for j = 1:length(b)
            if !iszero(Ξ[j+length(b),i])
                if eq_n === nothing
                    eq_n = p[cnt]*b[j]
                else
                    eq_n += p[cnt]*b[j]
                end
                p_[cnt] = Ξ[j+length(b),i]
                cnt += 1
            end
        end

        push!(b_, -eq_n ./ eq_d)
    end
    b_, p_
end
