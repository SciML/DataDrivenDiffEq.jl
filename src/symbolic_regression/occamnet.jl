# OccamNet
abstract type AbstractProbabilityLayer end
abstract type AbstractOccam end

## Overload softmax and logsoftmax with Temperature
NNlib.softmax(x, T; dims = 1) = T > eps() ? softmax(x ./ T, dims = dims) : softmax(x, dims = dims)
NNlib.logsoftmax(x, T; dims =1) = T > eps() ? x ./ T .- log.(sum(exp, x ./ T, dims = dims)) : logsoftmax(x, dims)

"""
$(TYPEDEF)

Defines a basic `ProbabilityLayer` in which the parameters act as probabilities via the `softmax` function for an
array of functions.

# Fields
$(FIELDS)

"""
mutable struct ProbabilityLayer{F,W,T,A} <: AbstractProbabilityLayer
    "Basis"
    op::F
    "Weights"
    weight::W
    "Temperature"
    t::T
    "Arieties of the basis"
    arieties::A
    "Skip connection"
    skip::Bool
end

function ProbabilityLayer(inp::Int, f::Function, t::Real = 1.0; skip = false, init_w = ones)
    arieties = ariety(f, typeof(t))
    w = init_w(eltype(t), sum(arieties), inp)
    return ProbabilityLayer([f], w, t, [arieties], skip)
end

function ProbabilityLayer(inp::Int, out::Int, t::Real = 1.0; skip = false, init_w = ones)
    arieties = [out]
    w = init_w(eltype(t), out, inp)
    return ProbabilityLayer(identity, w, t, arieties, skip)
end

function ProbabilityLayer(inp::Int, f::AbstractVector{Function}, t::Real = 1.0; skip = false, init_w = ones)
    arieties = [ariety(fi,typeof(t)) for fi in f]
    w = init_w(eltype(t), sum(arieties), inp)
    return ProbabilityLayer(f, w, t, arieties, skip)
end

function (p::AbstractProbabilityLayer)(x::AbstractVector)
     h̃ = softmax(p.weight, p.t, dims = 2)*x
     idx = 1
     res = map(1:length(p.arieties)) do i
         # Index of input
         args_ = h̃[idx:idx+p.arieties[i]-1]
         idx += p.arieties[i]
         p.op[i](args_...)
     end
     !p.skip && return res
     [res;x]
end

function (p::AbstractProbabilityLayer)(x::AbstractVector, sample::Vector{Int64})
    h̃ = [x[i] for i in sample]
    idx = 1
    res = map(1:length(p.arieties)) do i
        # Index of input
        args_ = h̃[idx:idx+p.arieties[i]-1]
        idx += p.arieties[i]
        p.op[i](args_...)
    end
    !p.skip && return res
    [res;x]
end

function (p::ProbabilityLayer{typeof(identity), W, T, A})(x::AbstractVector) where {W, T, A}
    res = softmax(p.weight, p.t, dims = 2)*x
    !p.skip && return res
    [res;x]
end

function (p::ProbabilityLayer{typeof(identity), W, T, A})(x::AbstractVector, sample::Vector{Int64}) where {W, T, A}
    res = [x[i] for i in sample]
    !p.skip && return res
    [res;x]
end

(p::AbstractProbabilityLayer)(x::AbstractMatrix) = hcat(map(p, eachcol(x))...)

"""
$(SIGNATURES)

Return the probability associated with the object by applying `softmax` on the weights.
"""
probabilities(p::AbstractProbabilityLayer) = softmax(p.weight, p.t, dims = 2)

function probabilities(p::AbstractProbabilityLayer, sample::Vector{Int})
    probs = probabilities(p)
    map(i->getindex(probs, i, sample[i]), 1:length(sample))
end

"""
$(SIGNATURES)

Return the logprobability associated with the object by applying `logsoftmax` on the weights.
"""

logprobabilities(p::AbstractProbabilityLayer) = logsoftmax(p.weight, p.t, dims = 2)

function logprobabilities(p::AbstractProbabilityLayer, sample::Vector{Int})
    logprobs = logprobabilities(p)
    map(i->getindex(logprobs, i, sample[i]), 1:length(sample))
end

# The resulting logprob
function logprobability(p::AbstractProbabilityLayer, sample, x)
    h̃ = logprobabilities(p, sample) + [x[i] for i in sample] # Add the input logprob to the output
    idx = 1
    res = map(1:length(p.arieties)) do i
        # Index of input
        logp = sum(h̃[idx:idx+p.arieties[i]-1])
        idx += p.arieties[i]
        logp
    end
    !p.skip && return res
    [res;x]
end


function logprobability(p::ProbabilityLayer{typeof(identity), W, T, A}, sample, x) where {W,T,A}
    h̃ = logprobabilities(p, sample) + [x[i] for i in sample]
    !p.skip && return h̃
    [h̃;x]
end

"""
$(SIGNATURES)

Return the probability of a pathway defined by sample associated with the object by applying `softmax` on the weights.
"""
resprobability(p::AbstractProbabilityLayer, sample) = exp.(reslogprobability(p, sample))

"""
$(SIGNATURES)

Set the temperature of the layer or net.
"""
set_temp!(p::AbstractProbabilityLayer, t::Real) = p.t = t

Base.rand(p::AbstractProbabilityLayer) = [rand(Categorical(Vector(w))) for w in eachrow(probabilities(p))]

function Base.show(io::IO, l::ProbabilityLayer)
  print(io, "ProbabilityLayer(", size(l.weight, 2), ", ", size(l.weight, 1))
  print(io, ", $(l.op)")
  l.skip ? print(io, ", Skip") : nothing
  print(io, ")")
end

Flux.@functor ProbabilityLayer
Flux.trainable(u::ProbabilityLayer) = (u.weight,)

## NETWORK

"""
$(TYPEDEF)

Defines a `OccamNet` which learns symbolic expressions from data using a probabalistic approach.
See [Interpretable Neuroevolutionary Models for Learning Non-Differentiable Functions and Programs
](https://arxiv.org/abs/2007.10784) for more details.

It get constructed via

```julia
net = OccamNet(inp::Int, outp::Int, layers::Int, f::Vector{Function}, t::Real = 1.0; constants = typeof(t)[], parameters::Int = 0, skip::Bool = false, init_w = ones, init_p = Flux.glorot_uniform)
```

`inp` describes the size of the input domain, `outp` the size of the output domain, `layers` the number of layers (including the input layer and excluding the linear output layer) and
`f` the functions to be used. Optional is the temperature `t` which is set to `1.0` at the beginning.

Keyworded arguments are `constants`, a vector of constants like π, ℯ which can concanated to the input, the number of trainable `parameters` and if `skip` connections should be used.
The constructors to the weights and parameters can be passed in via `init_w` and `init_p`.

`OccamNet` is callable with and without a specific route, which can be sampled from the networks weights via `rand(net)`.

# Fields
$(FIELDS)

"""
mutable struct OccamNet{F, C, P} <: AbstractOccam
    c::F # The chain
    constants::C # Additional constants which are fixed
    parameters::P # Additional learnable parameters
end

function OccamNet(inp::Int, outp::Int, layers::Int, f::Vector{Function}, t::Real = 1.0; constants = typeof(t)[], parameters::Int = 0, skip::Bool = false, init_w = ones, init_p = Flux.glorot_uniform)
    inp += parameters
    inp += length(constants)
    x0 = rand(typeof(t), inp)
    ls = []
    for l in 1:layers
        l_ = ProbabilityLayer(inp, f, t, skip = skip, init_w = init_w)
        push!(ls, l_)
        x0 = l_(x0)
        inp = length(x0)
    end
    # Output layer
    push!(ls, ProbabilityLayer(inp, outp, t, skip = false, init_w = init_w))
    b = parameters > 0 ? convert.(typeof(t), init_p(parameters)) : typeof(t)[]
    return OccamNet(Chain(ls...),constants, b)
end

function Base.show(io::IO, l::OccamNet)
  print(io, "OccamNet(", length(l.c))
  !isempty(l.constants) ? print(io, ", Constants ", length(l.constants)) : nothing
  !isempty(l.constants) ? print(io, ", Parameters ", length(l.parameters)) : nothing
  print(io, ")")
end

(o::OccamNet)(x) = o.c([x;o.constants;o.parameters])
(o::OccamNet)(x::AbstractMatrix) = hcat(map(o, eachcol(x))...)
set_temp!(o::OccamNet, t::Real) = map(x->set_temp!(x, t), o.c)
probabilities(o::OccamNet) = map(probabilities, o.c)
probabilities(o::OccamNet, route) = map(probabilities, o.c, route)

logprobabilities(o::OccamNet) = map(logprobabilities, o.c)
logprobabilities(o::OccamNet, route) = map(logprobabilities, o.c, route)

Base.rand(o::OccamNet) = map(rand, o.c)

function (o::OccamNet)(x::AbstractVector, route::Vector{Vector{Int64}})
    res = [x;o.constants;o.parameters]
    for i in 1:length(route)
        res = o.c[i](res,route[i])
    end
    return res
end

(o::OccamNet)(x::AbstractMatrix, route::Vector{Vector{Int64}}) = hcat(map(xi->o(xi, route), eachcol(x))...)

function logprobability(o::OccamNet, route::Vector{Vector{Int64}})
    # Input prob = 1 -> logprob = 0
    res = zeros(eltype(o.c[1].t), size(o.c[1].weight, 2) + length(o.constants))
    for i in 1:length(route)
        res = logprobability(o.c[i], route[i], res)
    end
    return res
end

probability(o::OccamNet, route::Vector{Vector{Int64}}) = exp.(logprobability(o, route))

Flux.@functor OccamNet
Flux.trainable(u::OccamNet) = (u.parameters, Flux.trainable.(u.c)...,)

gkernel(x,y,σ = one(eltype(x))) = 1/sqrt(2π*σ^2) .* exp.(-(x-y).^2 ./(2*σ))

function gkernel(x::AbstractMatrix, y::AbstractMatrix, σ::AbstractVector = ones(eltype(x), size(x, 1)))
    vcat(map(i->gkernel(x[i,:], y[i,:], σ[i])', 1:size(x, 1))...)
end

"""
$(SIGNATURES)

Overloads `Flux.train!` method to be used with an `OccamNet`.
"""
function Flux.train!(net::OccamNet, X, Y, opt, maxiters = 10; routes = 10, nbest = 1, cb = ()->(), progress = false)
    ny = size(Y,1)
    vary = [var(Y, dims = 2)...]
    ps_prob = Flux.params(net)

    if progress
        prog = Progress(
            maxiters, "Training $(net)"
        )
    end

    for k in 1:maxiters
        ls = map(1:routes) do i
            route = rand(net)
            res = net(X, route)
            route, sum(gkernel(res, Y, vary))
        end
        ls = sort(ls, by = last, rev = true)

        candidates = first.(ls[1:nbest])

        # Update probabilities
        gs = Flux.gradient(ps_prob) do
            l_ = zero(eltype(X))
            for c in candidates
                res = net(X, c)
                p = logprobability(net, c)
                l_ += dot(p, sum(gkernel(res, Y, vary), dims =2))
            end
            -l_
        end
        Flux.Optimise.update!(opt, ps_prob, gs)

        if progress
            rp = round.(exp.(logprobability(net, first(first(ls)))), digits = 5)
            loss = sum(abs2, net(X, first(first(ls))) - Y) / size(Y, 2)

            ProgressMeter.next!(
            prog;
            showvalues = [
                (:Probabilities, rp), (Symbol("Equivalent L2-Loss"), loss)
                ]
                )
        end
    end
    return
end

## SR WRAPPER

"""
$(TYPEDEF)

Options for using OccamNet within the `solve` function. Automatically creates a network with the given specification.

# Fields
$(FIELDS)

"""
struct OccamSR{F, C, T} <: AbstractSymbolicRegression "Functions used within the network"
    functions::F
    "Constants added to the input"
    constants::C
    "Number of layers"
    layers::Int
    "Number of parameters"
    parameters::Int
    "Activate skip connections"
    skip::Bool
end

function Base.show(io::IO, l::OccamSR{F, C, T}) where {F,C,T}
  T ? print(io, "Implicit ") : nothing
  print(io, "OccamSR(", length(l.functions))
  !isempty(l.constants) ? print(io, ", Constants ", length(l.constants)) : nothing
  l.parameters > 0 ? print(io, ", Parameters ", l.parameters) : nothing
  print(io, ")")
end

function Base.summary(io::IO, l::OccamSR)
  print(io, "OccamSR\n")
  print(io, "Functions ", l.functions, "\n")
  !isempty(l.constants) ? print(io, "Constants ", l.constants, "\n") : nothing
  l.parameters > 0 ? print(io, "Parameters ", l.parameters, "\n") : nothing
end

function Base.print(io::IO, l::OccamSR)
    summary(io, l)
end

function OccamSR(;functions = [+,*,sin,exp], constants = [π, ℯ],
    layers = 2, parameters = 0, skip = true, implicit = false,
    kwargs...)
    implicit && throw(error("Implicit OccamSR is not supported at the moment."))
    return OccamSR{typeof(functions), typeof(constants), implicit}(functions, constants, layers, parameters, skip)
end

## SOLVE

function DiffEqBase.solve(
    p, o::OccamSR{F,C, false}, opt;
    max_iter = 1000, cb = ()->(), progress = false, routes = 10, nbest = 1, temperature = 1.0
    ) where {F,C}

    # Target variables
    Y = get_target(p)
    # Inputs
    X̂, _, t, U = get_oop_args(p)

    t = iszero(t) ? [] : t
    # Cat the inputs
    X = vcat([x for x in (X̂, U, permutedims(t)) if !isempty(x)]...)

    inp = size(X,1)
    outp = size(Y,1)

    net = OccamNet(inp, outp, o.layers, o.functions, temperature, constants = o.constants, parameters = o.parameters, skip = o.skip)

    Flux.train!(net, X, Y, opt, max_iter, routes = routes, nbest = nbest, cb = cb, progress = progress)
    build_solution(p, net, o, opt)
end

## SOLUTION
function build_solution(prob::DataDrivenProblem, net::OccamNet, o::OccamSR, opt;
    eval_expression = false)

    @variables x[1:size(prob.X, 1)] u[1:size(prob.U,1)] t
    x_ = [x;u;t]

    inp = size(net.c[1].weight, 2)- length(net.constants)
    temp_ = net.c[1].t

    # Draw routers
    set_temp!(net, 0.1)
    route = rand(net)
    eqs = simplify.(net(x_[1:inp], route))
    set_temp!(net, temp_)


    # Build the lhs
    if length(eqs) == size(prob.X, 1)
        d = Differential(t)
        eqs = [d(x[i]) ~ eq for (i,eq) in enumerate(eqs)]
    end

    # Build a basis
    res_ = Basis(
        eqs, x, iv = t,
        controls = u,
        eval_expression = eval_expression
    )



    X = get_target(prob)
    Y = res_(get_oop_args(prob)...)

    # Build the metrics
    pb = exp(sum(logprobability(net, route)))
    pbs = probability(net, route)
    retcode = pb > 0.5 ? :sucess : :unlikely

    error = norm(X-Y, 2)
    k = free_parameters(res_)
    aic = AICC(k, X, Y)
    errors = zeros(eltype(X), size(Y, 1))
    aiccs = zeros(eltype(X), size(Y, 1))
    j = 1
    for i in 1:size(Y,1)

        errors[i] = norm(X[i,:].-Y[i,:],2)
        aiccs[i] = AICC(k, X[i:i, :], Y[i:i,:])
    end

    metrics = (
        Probability = pb,
        Error = error,
        AICC = aic,
        Probabilities = pbs,
        Errors = errors,
        AICCs = aiccs,
    )

    inputs = (
        Problem = prob,
        Algorithm = o,
    )

    return DataDrivenSolution(
        res_, retcode, [], opt, net, inputs, metrics
    )
end
