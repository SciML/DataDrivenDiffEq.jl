# TODO I think here is some potential for faster computation
# However, up to 25 states, the algorithm works fine and fast (main knobs are rtol and maxiter)
# This is the specialized version assuming a mass matrix form / linear in dx
# M(x, p)*dx = f(x, p)

# TODO preallocation
function ISindy(X, Ẋ, Ψ::Basis; maxiter::Int64 = 10, rtol::Float64 = 0.99, p::AbstractArray = [], opt::T = ADM()) where T <: DataDrivenDiffEq.Optimise.AbstractSubspaceOptimiser
    nb = length(Ψ.basis)

    # Compute the library and the corresponding nullspace
    θ = hcat([Ψ(xi, p = p) for xi in eachcol(X)]...)
    # Init for sweep over the differential variables
    eqs = Operation[]

    @inbounds for i in 1:size(Ẋ, 1)
        dθ = hcat(map((dxi, ti)->dxi.*ti, Ẋ[i, :], eachcol(θ))...)
        Θ = vcat(dθ, θ)
        N = nullspace(Θ', rtol = rtol)
        Q = deepcopy(N) # Deepcopy for inplace

        # Find sparse vectors in nullspace
        # Calls effectively the ADM algorithm with varying initial conditions
        DataDrivenDiffEq.fit!(Q, N', opt, maxiter = maxiter)


        # Compute pareto front
        pareto = map(q->norm(norm(q, 0) + norm(Θ'*q, 2), 2), eachcol(Q))
        score, posmin = findmin(pareto)
        # Get the corresponding eqs
        q_best = Q[:, posmin]
        # Remove small entries
        q_best[abs.(q_best) .< opt.λ] .= zero(eltype(q_best))
        # TODO Should be scaled to one, but does not seem to be
        rmul!(q_best ,one(eltype(q_best))/maximum(abs.(q_best)))

        # Numerator and Denominator
        # Maybe there is a better way of doing this
        Fn = simplify_constants(sum(q_best[nb+1:end].*Ψ.basis))
        Fd = simplify_constants(sum(q_best[1:nb].*Ψ.basis))

        push!(eqs, -Fn/Fd)
    end

    return Basis(eqs, variables(Ψ))
end
