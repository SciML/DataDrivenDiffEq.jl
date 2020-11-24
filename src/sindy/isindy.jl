# TODO I think here is some potential for faster computation
# However, up to 25 states, the algorithm works fine and fast (main knobs are rtol and maxiter)
# This is the specialized version assuming a mass matrix form / linear in dx
# M(x, p)*dx = f(x, p)
# Where M is diagonal!

# TODO preallocation

"""
    ISINDy(X, Y, Ψ, opt = ADM(); f, g, maxiter, rtol, p, t, convergence_error)
    ISINDy(X, Y, Ψ, opt; f, g, maxiter, rtol, p, t, convergence_error, normalize, denoise)
    ISINDy(X, Y, Ψ, lamdas, opt; f, g, maxiter, rtol, p, t, convergence_error, normalize, denoise)

Performs an implicit sparse identification of nonlinear dynamics given the data matrices `X` and `Y` via the `AbstractBasis` `basis.`
Keyworded arguments include the parameter (values) of the basis `p` and the timepoints `t`, which are passed in optionally.
Tries to find a sparse vector inside the nullspace if `opt` is an `AbstractSubspaceOptimizer` or performs parallel implicit search if `opt` is a `AbstractOptimizer`.
`maxiter` denote the maximum iterations to perform and `convergence_error` the
bound which causes the optimizer to stop. `denoise` defines if the matrix holding candidate trajectories should be thresholded via the [optimal threshold for singular values](http://arxiv.org/abs/1305.5870).
`normalize` normalizes the matrix holding candidate trajectories via the L2-Norm over each function.

Typically `X` represent the state measurements and `Y` the measurements of the differential state. Since only the number of measurements (column dimension of the matrices) have to be equal, it is possible to augment `X` with additional
data, e.g. external forcing or inputs.

If `ISINDy` is called with an additional array of thresholds contained in `lambdas`, it performs a multi objective optimization over all thresholds.
The best vectors of the sparse nullspace are selected via multi-objective optimization.
The best candidate is determined via the mapping onto a feature space `f` and an (scalar, positive definite) evaluation `g`.
The signature of should be `f(xi, theta)` where `xi` are the coefficients of the sparse optimization and `theta` is the evaluated candidate library.
`rtol` gets directly passed into the computation of the nullspace.

Currently ISINDy supports functions of the form `g(u, p, t)*du - f(u, p, t) = 0`.

Returns a `SparseIdentificationResult`.
"""
function ISINDy(X::AbstractArray, Ẋ::AbstractArray, Ψ::Basis, opt::T = ADM(); f::Function = (xi, theta)->[norm(xi, 0); norm(theta'*xi, 2)], g::Function = x->norm(x), maxiter::Int64 = 10, rtol::Float64 = 0.99, p::AbstractArray = [], t::AbstractVector = [], convergence_error = eps()) where T <: DataDrivenDiffEq.Optimize.AbstractSubspaceOptimizer
    @assert size(X)[end] == size(Ẋ)[end]

    # Compute the library and the corresponding nullspace
    θ = zeros(eltype(X), length(Ψ), size(X, 2))
    Ψ(θ, X, p, t)

    # Init for sweep over the differential variables
    Ξ = zeros(eltype(θ), length(Ψ)*2, size(Ẋ, 1))

    iters = Optimize.fit!(Ξ, θ, Ẋ, opt, maxiter = maxiter, rtol = rtol, convergence_error = convergence_error, f = f, g = g)

    return ImplicitSparseIdentificationResult(Ξ, Ψ, iters , opt, iters <= maxiter, Ẋ, X, p = p, t = t)
end


function ISINDy(X::AbstractArray, Ẋ::AbstractArray, Ψ::Basis, opt::T; f::Function = (xi, theta)->[norm(xi, 0); norm(theta'*xi, 2)], g::Function = x->norm(x), maxiter::Int64 = 10, rtol::Float64 = 0.99, p::AbstractArray = [], t::AbstractVector = [], convergence_error = eps(), normalize::Bool = true, denoise::Bool = false) where T <: DataDrivenDiffEq.Optimize.AbstractOptimizer
    @assert size(X)[end] == size(Ẋ)[end]

    # Compute the library and the corresponding nullspace
    θ = zeros(eltype(X), length(Ψ), size(X, 2))
    Ψ(θ, X, p, t)

    dθ = zeros(eltype(θ), size(θ, 1)*2, size(θ, 2))
    dθ[size(θ, 1)+1:end, :] .= θ

    # Init for sweep over the differential variables
    Ξ = zeros(eltype(θ), length(Ψ)*2, size(Ẋ, 1))
    q = zeros(eltype(θ), size(θ, 1)*2, size(θ, 1)*2)
    
    # Closure
    fg(xi, theta) = (g∘f)(xi, theta)

    iters = zeros(Int64, size(Ẋ, 1))
    

    # TODO maybe add normalization here
    for i in 1:size(Ẋ, 1)
        for j in 1:size(θ, 2)
            dθ[1:size(θ, 1), j] .= Ẋ[i, j].*@view(θ[:, j])
        end

        iters[i] = parallel_implicit!(view(Ξ,:, i), view(dθ, :, :), opt, fg, maxiter, denoise, normalize, convergence_error)

    end

    return ImplicitSparseIdentificationResult(Ξ, Ψ, minimum(iters) , opt, minimum(iters) <= maxiter, Ẋ, X, p = p, t = t)
end


function ISINDy(X::AbstractArray, Ẋ::AbstractArray, Ψ::Basis, thresholds::AbstractVector, opt::T = STRRidge(); f::Function = (xi, theta)->[norm(xi, 0); norm(theta'*xi, 2)], g::Function = x->norm(x), maxiter::Int64 = 10, rtol::Float64 = 0.99, p::AbstractArray = [], t::AbstractVector = [], convergence_error = eps(), normalize::Bool = true, denoise::Bool = false) where T <: DataDrivenDiffEq.Optimize.AbstractOptimizer
    @assert size(X)[end] == size(Ẋ)[end]
    
    # Compute the library and the corresponding nullspace
    θ = zeros(eltype(X), length(Ψ), size(X, 2))
    Ψ(θ, X, p, t)
    
    dθ = zeros(eltype(θ), size(θ, 1)*2, size(θ, 2))
    dθ[size(θ, 1)+1:end, :] .= θ

    # Init for sweep over the differential variables
    Ξ = zeros(eltype(θ), length(Ψ)*2, size(Ẋ, 1))

    q = zeros(eltype(θ), size(θ, 1)*2, size(θ, 1)*2)
    # Closure
    fg(xi, theta) = (g∘f)(xi, theta)

    iters = zeros(Int64, size(Ẋ, 1))
    

    # TODO maybe add normalization here
    for i in 1:size(Ẋ, 1)
        for j in 1:size(θ, 2)
            dθ[1:size(θ, 1), j] .= Ẋ[i, j].*@view(θ[:, j])
        end

        iters[i] = parallel_implicit!(view(Ξ,:, i), view(dθ, :, :), opt, fg, maxiter, denoise, normalize, convergence_error, thresholds)

    end

    return ImplicitSparseIdentificationResult(Ξ, Ψ, minimum(iters) , opt, minimum(iters) <= maxiter, Ẋ, X, p = p, t = t)
end

function parallel_implicit!(Ξ, X, opt, fg, maxiter, denoise, normalize, convergence_error, thresholds = [get_threshold(opt)])

    q = zeros(eltype(X), size(X, 1), size(X, 1), length(thresholds))
    iters_ = zeros(Int64, size(X,1), length(thresholds))
    
    Threads.@threads for j in 1:size(X, 1)
        idx = [k_ != j for k_ in 1:size(X, 1)]
        
        for k in 1:length(thresholds)
            set_threshold!(opt, thresholds[k])
            iters_[j, k] = sparse_regression!(view(q, idx, j, k), view(X, idx , :), -transpose(view(X, j, :)), maxiter , opt, denoise, normalize, convergence_error) 
            q[j, j, k] = one(eltype(q))
        end
    end
        
    iters = Inf
    half_size = Int64(round(size(X, 1)/2))
    
    for j in 1:size(X, 1), k in 1:length(thresholds)
        norm(view(q, 1:half_size , j, k), 0) <= 0 || norm(view(q, (half_size+1):size(X, 1),j, k), 0)<= 0 ? continue : nothing
        
        if evaluate_pareto!(view(Ξ, :), view(q, :, j, k), fg, view(X, :, :)) || j == 1
            @views mul!(Ξ, q[:, j, k],  one(eltype(q))./maximum(abs, q[:, j, k]))

            iters_[j, k] < iters ? iters = iters_[j, k] : nothing
        end
    end

    return iters
end


function ImplicitSparseIdentificationResult(coeff::AbstractArray, equations::Basis, iters::Int64, opt::T, convergence::Bool, Y::AbstractVecOrMat, X::AbstractVecOrMat; p::AbstractArray = [], t::AbstractVector = []) where T <: Union{Optimize.AbstractOptimizer, Optimize.AbstractSubspaceOptimizer}

    sparsities = Int64.(norm.(eachcol(coeff), 0))

    b_, p_ = derive_implicit_parameterized_eqs(coeff, equations)
    ps = [p; p_]

    Ŷ = b_(X, ps, t)
    training_error = [norm(Y[i, :]-Ŷ[i,:], 2) for i in 1:size(Ŷ, 1)]
    aicc = similar(training_error)

    for i in 1:length(aicc)
        aicc[i] = AICC(sum(sparsities[i]), view(Ŷ, i, :) , view(Y, i, :))
    end

    return SparseIdentificationResult(coeff, [p...;p_...], b_ , opt, iters, convergence,  training_error, aicc,  sparsities)
end


function derive_implicit_parameterized_eqs(Ξ::AbstractArray{T, 2}, b::Basis) where T <: Real
    
    sparsity = Int64(norm(Ξ, 0)) # Overall sparsity
    @parameters p[1:sparsity]
    size_b = length(b)

    inds = @. ! iszero(Ξ)

    pinds = Int64.(norm.(eachcol(inds), 0))
    
    pinds_d = Int64.(norm.(eachcol(inds[1:size_b,:]), 0))
    pinds_n = Int64.(norm.(eachcol(inds[size_b+1:end, :]), 0))
    
    p_ = similar(Ξ[inds])

    eq = Array{Any}(undef, sum([i>0 for i in pinds]))
    eq .= 0

    cnt = 1
    p_cnt = 1
    eq_n = 0
    eq_d = 0

    @views for i=1:size(Ξ, 2)
        # Numerator
        eq_n = 0
        eq_d = 0
        if iszero(pinds_n[i]) || iszero(pinds_d[i])
            continue
        else
            for j in 1:size_b
                if inds[j, i]
                    if iszero(eq_d)
                        eq_d =  p[p_cnt] * (b.eqs[j]).rhs
                    else    
                        eq_d +=  p[p_cnt] * (b.eqs[j]).rhs
                    end
                    p_[p_cnt] = Ξ[j, i]
                    p_cnt += 1
                end

                if inds[j+size_b, i]
                    if iszero(eq_n)
                        eq_n =  p[p_cnt] * (b.eqs[j]).rhs
                    else
                        eq_n +=  p[p_cnt] * (b.eqs[j]).rhs
                    end
                    p_[p_cnt] = Ξ[j+size_b, i]
                    p_cnt += 1
                end
            end

            eq[cnt] = - eq_n / eq_d
        end
        cnt += 1
    end

    b_ = Basis(eq, variables(b), parameters = vcat(parameters(b), p), iv = independent_variable(b))

    b_, p_
end
