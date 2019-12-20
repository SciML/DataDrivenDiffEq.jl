# Simple ridge regression based upon the sindy-mpc
# repository, see https://arxiv.org/abs/1711.05501
# and https://github.com/eurika-kaiser/SINDY-MPC/blob/master/LICENSE
function STRridge(A::AbstractArray, Y::AbstractArray; ϵ::Number = 1e-3, maxiter::Int64 = 100)
    # Initial guess
    Ξ = A \ Y
    for i in 1:maxiter
        smallinds = abs.(Ξ) .<= ϵ
        Ξ[smallinds] .= 0.0
        for (j, y) in enumerate(eachcol(Y))
            biginds = @. ! smallinds[:, j]
            Ξ[biginds, j] = A[:, biginds] \ y
        end
    end
    return Ξ
end

function sparseConvex(A::AbstractArray, Y::AbstractArray; ϵ::Float64 = 1e-3)
    # Performs ~5 times faster than STRRidge
    # Get the getdimensions
    n_states, n_measurements = size(Y)
    n_basis = size(A)[2]
    # Preallocate the weights
    Ξ = zeros(n_basis, n_states)

    # Initialize the mixed integer programming
    t = Convex.Variable(n_basis)
    x = Convex.Variable(n_basis)
    dx = Convex.Variable(n_measurements)
    # Has always the same form
    p = Convex.minimize(Convex.dot(ones(n_basis),t))
    p.constraints += A*x == dx
    p.constraints += x <= t
    p.constraints += x >= -t

    # Iterate over all variables
    for i in 1:n_states
        # Fix the value for dx
        dx.value = Y[i,:]
        Convex.fix!(dx)
        # Solve
        Convex.solve!(p, GLPKMathProgInterface.GLPKSolverMIP())
        # TODO Warmstarting the solver is not an option right now
        #, warmstart = i > 1 ? true : false, verbose = false)
        Ξ[:, i] = x.value
    end

    Ξ[abs.(Ξ) .<= ϵ] .= 0.0

    return Ξ'
end

function simplified_matvec(Ξ, basis)
    eqs = Operation[]
    for i=1:size(Ξ, 1)
        eq = nothing
        for j = 1:size(Ξ, 2)
            if !iszero(Ξ[i,j])
                if eq === nothing
                    eq = basis[j]*Ξ[i,j]
                else
                    eq += basis[j]*Ξ[i,j]
                end
            end
        end
        if eq === nothing
            push!(eqs, zero(basis[1]))
        else
            push!(eqs, eq)
        end
    end
    eqs
end

# Returns a basis for the differential state
function SInDy(X::AbstractArray, Ẋ::AbstractArray, Ψ::Basis; p::AbstractArray = [], ϵ::Number = 1e-1)
    θ = hcat([Ψ(xi, p = p) for xi in eachcol(X)]...)
    Ξ = sparseConvex(θ', Ẋ, ϵ = ϵ)
    return Basis(simplified_matvec(Ξ, Ψ.basis), variables(Ψ), parameters = p)
end
