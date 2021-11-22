"""
$(TYPEDEF)

The solution to a `DataDrivenProblem` derived via a certain algorithm.
The solution is represented via an `AbstractBasis`, which makes it callable.

# Fields
$(FIELDS)

# Note

The L₂ norm error, AIC and coefficient of determinantion get only computed, if eval_expression is set to true or
if the solution can be interpreted as a linear regression result.
"""
struct DataDrivenSolution{L, A, O} <: AbstractDataDrivenSolution
    "The basis representation of the solution"
    basis::AbstractBasis
    "Parameters of the solution"
    parameters::AbstractVecOrMat
    "Returncode"
    retcode::Symbol
    "Algorithm"
    alg::A
    "Original output of the solution algorithm"
    out::O
    "Problem"
    prob::AbstractDataDrivenProblem
    "L₂ norm error"
    l2_error::AbstractVector
    "AIC"
    aic::AbstractVector
    "Coefficient of determinantion"
    rsquared::AbstractVector

    function DataDrivenSolution(linearity::Bool,b::AbstractBasis, p::AbstractVector, retcode::Symbol, alg::A, out::O, prob::AbstractDataDrivenProblem; kwargs...) where {A,O}
        return new{linearity, A,O}(
            b, p, retcode, alg, out, prob
        )
    end

    function DataDrivenSolution(linearity::Bool,b::AbstractBasis, p::AbstractVector, retcode::Symbol, alg::A, out::O, prob::AbstractDataDrivenProblem, l2::AbstractVector; kwargs...) where {A,O}
        return new{linearity, A,O}(
            b, p, retcode, alg, out, prob, l2
        )
    end

    function DataDrivenSolution(linearity::Bool,b::AbstractBasis, p::AbstractVector, retcode::Symbol, alg, out, prob::AbstractDataDrivenProblem, l2::AbstractVector, aic::AbstractVector; kwargs...)
        return new{linearity, typeof(alg), typeof(out)}(
            b, p, retcode, alg, out, prob, l2, aic
        )
    end

    function DataDrivenSolution(b::AbstractBasis, p::AbstractVector, retcode::Symbol, alg, out, prob::AbstractDataDrivenProblem, l2::AbstractVector, aic::AbstractVector, rsquared::AbstractVector; kwargs...)
        return new{true, typeof(alg), typeof(out)}(
            b, p, retcode, alg, out, prob, l2, aic, rsquared
        )
    end

        
end

function DataDrivenSolution(b::AbstractBasis, p::AbstractVector, retcode::Symbol, alg::A, out::O, prob::AbstractDataDrivenProblem, linearity::Bool; 
    eval_expression = false, kwargs...) where {A,O}
        
    if !eval_expression
        # Compute the errors
        x, _, t, u = get_oop_args(prob)
        e = get_target(prob) - b(x, p, t, u)

        l2 = sum(abs2, e, dims = 2)[:,1]
        aic = 2*(-size(e, 2) .* log.(l2 / size(e, 2)) .+ length(p))
        
        if linearity
            rsquared = 1 .- mean(e, dims = 2)[:,1] ./ var(get_target(prob), dims = 2)[:,1]
            #return l2, aic, rsquared
            return DataDrivenSolution(
                b, p, retcode, alg, out, prob, l2, aic, rsquared
            )
        end

        return DataDrivenSolution(
            linearity, b, p, retcode, alg, out, prob, l2, aic
        )
    end

    return DataDrivenSolution(
        linearity, b, p, retcode, alg, out, prob
    )
end



## Make it callable
(r::DataDrivenSolution)(args...) = r.basis(args...)


function Base.show(io::IO, ::DataDrivenSolution{linearity}) where linearity
    if linearity
        print(io, "Linear Solution")
    else
        print(io, "Nonlinear Solution")
    end
end

function Base.print(io::IO, r::DataDrivenSolution{linearity, a, o}) where {linearity, a, o}
    show(io, r)
    print(io, " with $(length(r.basis)) equations and $(length(r.parameters)) parameters.\n")
    print(io, "Returncode: $(r.retcode)\n")
    isdefined(r, :l2_error) && print(io, "L₂ Norm error : $(r.l2_error)\n")
    isdefined(r, :aic) && print(io,"AIC : $(r.aic)\n")
    isdefined(r, :rsquared) && print(io, "R² : $(r.rsquared)\n")
    return
end


function Base.print(io::IO, r::DataDrivenSolution, fullview::DataType)

    fullview != Val{true} && return print(io, r)

    print(io, r)

    if length(r.parameters) > 0
        x = parameter_map(r)
        println(io, "Parameters:")
        for v in x
            println(io, "   $(v[1]) : $(v[2])")
        end
    end

    return
end





##
"""
$(SIGNATURES)

Returns the `Basis` of the result.
"""
result(r::DataDrivenSolution) = r.basis

"""
$(SIGNATURES)

Returns the AIC of the result.
"""
aic(r::DataDrivenSolution) = begin
    isdefined(r, :aic) && return r.aic
    return NaN
end

"""
$(SIGNATURES)

Returns the L₂ norm error of the result.
"""
error(r::DataDrivenSolution) = begin
    isdefined(r, :l2_error) && return r.l2_error
    return NaN
end

"""
$(SIGNATURES)

Returns the coefficient of determinantion of the result, if the result has been
derived via a linear regression, e.g. sparse regression or koopman.
"""
determination(r::DataDrivenSolution{l, o, a}) where {l,o, a} = begin
    if l
        return r.rsquared
    else
        return NaN
    end
end


"""
$(SIGNATURES)

Returns the estimated parameters in form of an `Vector`.
"""
ModelingToolkit.parameters(r::DataDrivenSolution) = r.parameters

"""
$(SIGNATURES)

Generate a mapping of the parameter values and symbolic representation useable
to `solve` and `ODESystem`.
"""
function parameter_map(r::DataDrivenSolution)
    return [
        ps_ => p_ for (ps_, p_) in zip(parameters(r.basis), r.parameters)
    ]
end


"""
$(SIGNATURES)

Returns the algorithm used to derive the solution.
"""
algorithm(r::DataDrivenSolution) = r.alg

"""
$(SIGNATURES)

Returns the original output of the algorithm.
"""
output(r::DataDrivenSolution) = r.out

"""
$(SIGNATURES)

Returns all applicable metrics of the solution.
"""
function metrics(r::DataDrivenSolution)
    fnames_ = (:l2_error, :aic, :rsquared)
    names_ = (:L₂, :AIC, :R²)
    m = Dict() 
    for i in 1:length(fnames_)
        if isdefined(r, fnames_[i]) 
            push!(m, names_[i] => getfield(r, fnames_[i]))
        end
    end
    m
end
## Helper for the solution

# Check linearity

function assert_linearity(eqs::AbstractVector{Equation}, x::AbstractVector{Num})
    return assert_linearity(map(x->Num(x.rhs), eqs), x)
end

# Returns true iff x is not in the arguments of the jacobian of eqs
function assert_linearity(eqs::AbstractVector{Num}, x::AbstractVector{Num})
    j = Symbolics.jacobian(eqs, x)
    # Check if any of the variables is in the jacobian
    v = unique(reduce(vcat, map(get_variables, j)))
    for xi in x, vi in v
        isequal(xi, vi) && return false
    end
    return true
end

function construct_basis(X, b, implicits = Num[]; dt = one(eltype(X)), lhs::Symbol = :continuous, is_implicit = false, eval_expression = false)

    # Create additional variables
    sp = Int(norm(X, 0))
    sps = norm.(eachcol(X), 0)
    inds = sps .> zero(eltype(X))
    pl = length(parameters(b))
    
    p = [Symbolics.variable(:p, i) for i in (pl+1):(pl+sp)]
    p = collect(p)
    ps = zeros(eltype(X), sp)

    eqs = zeros(Num, sum(inds))
    eqs_ = [e.rhs for e in equations(b)]
    cnt = 1
    for j in 1:size(X, 2)
        if sps[j] == zero(eltype(X))
            continue
        end
        for i in 1:size(X, 1)
            if iszero(X[i,j])
                continue
            end
            ps[cnt] = X[i,j]
            eqs[j] += p[cnt]*eqs_[i]
            cnt += 1
        end
    end

    # Build the lhs
    xs = states(b)
    if isempty(implicits) || !is_implicit
        if length(eqs) == length(states(b))
            if lhs == :continuous
                d = Differential(get_iv(b))
            elseif lhs == :discrete
                d = Difference(get_iv(b), dt = dt)
            end
            eqs = [d(xs[i]) ~ eq for (i,eq) in enumerate(eqs)]
        end
    else
        eqs = 0 .~ eqs
        if !isempty(implicits)
            if assert_linearity(eqs, implicits)
                # Try to solve the eq for the implicits
                eqs = ModelingToolkit.solve_for(eqs, implicits)
                eqs = implicits .~ eqs
            end
            xs = [s for s in xs if !any(map(i->isequal(i, s), implicits))]
        end
    end

    Basis(
        eqs, xs,
        parameters = [parameters(b); p], iv = get_iv(b),
        controls = controls(b), observed = observed(b),
        name = gensym(:Basis),
        eval_expression = eval_expression
    ), ps
end

function _round!(x::AbstractArray{T, N}, digits::Int) where {T, N}
    for i in eachindex(x)
        x[i] = round(x[i], digits = digits)
    end
    return x
end

function assert_lhs(prob)
    dt = mean(diff(prob.t))
    lhs = :direct
    if isa(prob, AbstracContProb)
        lhs = :continuous
    elseif isa(prob, AbstractDiscreteProb)
        lhs = :discrete
    else
        lhs = :direct
    end 
    return lhs, dt
end

function DataDrivenSolution(prob::AbstractDataDrivenProblem, Ξ::AbstractMatrix, opt::Optimize.AbstractOptimizer, b::Basis, implicits = Num[]; eval_expression = false, digits::Int = 10, kwargs...)
    # Build a basis and returns a solution
    if all(iszero.(Ξ))
        @warn "Sparse regression failed! All coefficients are zero."
        return DataDrivenSolution(
        b, [], :failed, opt, Ξ, prob)
    end
 
    # Assert continuity
    lhs, dt = assert_lhs(prob)

    sol , ps = construct_basis(round.(Ξ, digits = digits), b, implicits, 
        lhs = lhs, dt = dt,
        is_implicit = isa(opt, Optimize.AbstractSubspaceOptimizer) ,eval_expression = eval_expression
        )
    
    return DataDrivenSolution(
        sol, ps, :solved, opt, Ξ, prob, true, eval_expression = eval_expression
    )
end


function DataDrivenSolution(prob::AbstractDataDrivenProblem, k, C, B, Q, P, inds, b::AbstractBasis, alg::AbstractKoopmanAlgorithm; 
    digits::Int = 10, eval_expression = false, kwargs...)
    # Build parameterized equations, inds indicate the location of basis elements containing an input
    Ξ = zeros(eltype(B), size(C,2), length(b))


    Ξ[:, inds] .= real.(Matrix(k))
    if !isempty(B)
        Ξ[:, .! inds] .= B
    end

    # Assert continuity
    lhs, dt = assert_lhs(prob)
    
    b, ps = construct_basis(round.(C*Ξ, digits = digits)', b, 
        lhs = lhs, dt = dt,
        eval_expression = eval_expression)


    res_ = Koopman(equations(b), states(b),
        parameters = parameters(b),
        controls = controls(b), iv = get_iv(b),
        K = k, C = C, Q = Q, P = P, lift = b.f,
        is_discrete = is_discrete(prob),
        eval_expression = eval_expression)
    #return res_
    return DataDrivenSolution(
        res_, ps, :solved, alg, Ξ, prob, true, eval_expression = eval_expression
    )
end
