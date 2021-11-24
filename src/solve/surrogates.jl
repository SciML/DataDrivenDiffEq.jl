function entangle(f::Function, idx::Int)
    return f_(x) = getindex(f(x), idx)
end

function dependencies(∇, x::AbstractMatrix{T}, abstol = eps(), reltol= eps(); kwargs...) where T <: Number
    mean(map(xi->normalize(abs.(∇(xi)), Inf), eachcol(x))) .>= reltol
end

function linearities(∇, x::AbstractMatrix{T}, abstol = eps(), reltol= eps(); kwargs...) where T <: Number
    var(map(∇, eachcol(x))) .< reltol
end

mutable struct Surrogate <: AbstractSurrogate
    # Original function
    f::Function
    # Gradient 
    ∇::Function
    
    # Dependent variables
    deps::BitVector
    # Linear variables
    linears::BitVector

    # Split
    op::Function
    children::AbstractVector{Surrogate}

    function Surrogate(f, x::AbstractMatrix; abstol = eps(), reltol = eps(), kwargs...)
        y_ = f(x[:,1])
        if isa(y_, AbstractVector)
            return map(1:size(y_, 1)) do i
                f_ = entangle(f, i)
                Surrogate(f_, x, abstol = abstol, reltol = reltol; kwargs...)
            end
        end
        
        ∇(x) = ForwardDiff.gradient(f, x)
        
        deps = dependencies(∇, x, abstol, reltol; kwargs...)
        linears = linearities(∇, x, abstol, reltol; kwargs...)
        return new(
            f, ∇, deps, linears .* deps
        )
    end
end

has_children(s::Surrogate) = isdefined(s, :children)


"""
$(TYPEDEF)

Defines a composition of solvers to be applied on the specific surrogate function. 
'generator' should be defined in such a way that is generates basis elements for the (filtered) 
dependent variables for explicit and - if used - implicit sparse regression, e.g. 

```julia
generator(u) -> polynomial_basis(u,3)
generator(u,y)->vcat(generator(u), generator(u) .*y)

solvers = SurrogateSolvers(
    generator, STLSQ(), ImplicitOptimizer(); kwargs...
)
```

# Fields

$(FIELDS)

# Note
This is still an experimental prototype and highly sensitive towards the tolerances.
"""
struct SurrogateSolvers{G, A, K}
    generator::G
    alg::A
    kwargs::K

    function SurrogateSolvers(generator::Function, args...; kwargs...)
        return new{typeof(generator), typeof(args), typeof(kwargs)}(
            generator, args, kwargs
        )
    end
end

function linear_split!(s::Surrogate, x::AbstractMatrix, abstol = eps(), reltol = eps(); kwargs...)
    sum(s.linears) < 1 && return
    s.linears == s.deps && return
    has_children(s) && return


    # Get coefficients
    w = mean(map(s.∇, eachcol(x)))[s.linears]

    g = let w = w, idx = s.linears
        (x) -> dot(w, x[s.linears])
    end

    h = let f = s.f, g = g
        (x) -> f(x) - g(x)
    end

    setfield!(s, :op, +)
    setfield!(s, :children, 
        [
            Surrogate(g, x, abstol = abstol, reltol = reltol; kwargs...);
            Surrogate(h, x,  abstol = abstol, reltol = reltol; kwargs...)
        ]
    )
    return
end

function surrogate_solve(s::Surrogate, x::AbstractMatrix, sol::SurrogateSolvers; abstol = 1e-5, reltol = eps())
    has_children(s) && return composite_solve(s, x, sol, abstol = abstol, reltol = reltol)
    # First see if its a linear fit
    prob = DirectDataDrivenProblem(x, reduce(hcat, map(s.f, eachcol(x))))
    if s.linears == s.deps
        u = [Symbolics.variable("u", i) for i in 1:size(x,1)]
        ũ = u[s.deps]
        b = Basis([ũ; 1], u)
        return solve(prob, b, STLSQ(abstol); sol.kwargs...)
    end
    return pareto_solve(s, x, sol, abstol = abstol, reltol = reltol)
end

function pareto_solve(s::Surrogate, x::AbstractMatrix, sol::SurrogateSolvers; kwargs...)
    res = map(sol.alg) do ai
        surrogate_solve(s::Surrogate, x::AbstractMatrix, sol.generator, ai; kwargs..., sol.kwargs...)
    end
    valids = map(x->x.retcode == :solved, res)
    res = [res[i] for i in 1:length(valids) if valids[i]]
    idx = argmax(map(x->metrics(x)[:AIC], res))
    return res[idx]
end

function surrogate_solve(s::Surrogate, x::AbstractMatrix, g::Function, args...; kwargs...)
    prob = DirectDataDrivenProblem(x, reduce(hcat, map(s.f, eachcol(x))))
    return solve(prob, args...; kwargs...)
end

function surrogate_solve(s::Surrogate, x::AbstractMatrix, g::Function, opt::Optimize.AbstractOptimizer; kwargs...)
    prob = DirectDataDrivenProblem(x, reduce(hcat, map(s.f, eachcol(x))))
    u = [Symbolics.variable("u", i) for i in 1:size(x,1)]
    ũ = u[s.deps]
    b = Basis(g(ũ), u)
    return solve(prob, b, opt; kwargs...)
end

function surrogate_solve(s::Surrogate, x::AbstractMatrix, g::Function, opt::Optimize.AbstractSubspaceOptimizer; kwargs...)
    prob = DirectDataDrivenProblem(x, reduce(hcat, map(s.f, eachcol(x))))
    u = [Symbolics.variable("u", i) for i in 1:size(x,1)]
    y = Symbolics.variable("y")
    ũ = u[s.deps]
    b = Basis(g(ũ, y), [u;y])
    return solve(prob, b, opt, [y]; kwargs...)
end

function composite_solve(s::Surrogate, x::AbstractMatrix, sol::SurrogateSolvers; kwargs...)
    res = map(s.children) do c
        surrogate_solve(c, x, sol; kwargs...)
    end

    # Get the results
    basis = map(x->result(x), res)
    equations = map(x->first(x).rhs, ModelingToolkit.equations.(basis))
    pls = sum(length, parameters.(basis))
    p = [Symbolics.variable("p", i) for i in 1:pls]
    ps = reduce(vcat, map(parameters, res))
    eq = Num(0)
    cnt = 1
    for (i,ei) in enumerate(equations)
        subs = Dict()
        for p_ in parameters(basis[i])
            push!(subs, p_ => p[cnt])
            cnt += 1
        end
        eq = s.op(Num(eq), substitute(Num(ei), subs))
    end
    b_new = Basis([eq], Num.(states(first(basis))), parameters = p)
    prob = DirectDataDrivenProblem(x, reduce(hcat, map(s.f, eachcol(x))))
    return DataDrivenSolution(
        b_new, ps, :solved, map(x->x.alg, res), s.children, prob, true, eval_expression = true
    )
end


function DiffEqBase.solve(prob::DataDrivenProblem, f::Function, sol::SurrogateSolvers; abstol = eps(), reltol = eps(), kwargs...)
    x, _ = get_oop_args(prob)
    s = Surrogate(f, x, abstol = abstol, reltol = reltol)
    res = map(s) do si
        linear_split!(si, x, abstol = abstol, reltol = reltol)
        surrogate_solve(si, x, sol, abstol = abstol, reltol = reltol)
    end
end
