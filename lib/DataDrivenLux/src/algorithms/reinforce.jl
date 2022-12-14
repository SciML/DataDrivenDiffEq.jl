abstract type AbstractRewardScale{risk} end

"""
$(TYPEDEF)

Scales the losses in such a way that the minimum loss is equal to one.
"""
struct RelativeReward{risk} <: AbstractRewardScale{risk} end

RelativeReward(risk_seeking = true) = RelativeReward{risk_seeking}()

function (::RelativeReward)(losses::Vector{T}) where {T <: Number}
    exp.(minimum(losses) .- losses)
end

function (::RelativeReward{true})(losses::Vector{T}) where {T <: Number}
    r = exp.(minimum(losses) .- losses)
    r .- minimum(r)
end

"""
$(TYPEDEF)

Scales the losses in such a way that the minimum loss is the most influencial reward.
"""
struct AbsoluteReward{risk} <: AbstractRewardScale{risk} end

AbsoluteReward(risk_seeking = true) = AbsoluteReward{risk_seeking}()

function (::AbsoluteReward)(losses::Vector{T}) where {T <: Number}
    exp.(-losses)
end

function (::AbsoluteReward{true})(losses::Vector{T}) where {T <: Number}
    r = exp.(-losses)
    r .- minimum(r)
end
"""
$(TYPEDEF)

Uses the REINFORCE algorithm to search over the space of possible solutions to the 
symbolic regression problem.

# Fields
$(FIELDS)
"""
@with_kw struct Reinforce{F, A, L, O, R} <: AbstractDAGSRAlgorithm
    "Reward function which should convert the loss to a reward."
    reward::R = RelativeReward(false)
    "The number of candidates to track"
    populationsize::Int = 100
    "The functions to include in the search"
    functions::F = (sin, exp, cos, log, +, -, /, *)
    "The arities of the functions"
    arities::A = (1, 1, 1, 1, 2, 2, 2, 2)
    "The number of layers"
    n_layers::Int = 1
    "Include skip layers"
    skip::Bool = true
    "Evaluation function to sort the samples"
    loss::L = aicc
    "The number of candidates to keep in each iteration"
    keep::Union{Real, Int} = 0.1
    "Use protected operators"
    use_protected::Bool = true
    "Use distributed optimization and resampling"
    distributed::Bool = false
    "Use threaded optimization and resampling - not implemented right now."
    threaded::Bool = false
    "Random seed"
    rng::Random.AbstractRNG = Random.default_rng()
    "Optim optimiser"
    optimizer::O = LBFGS()
    "Optim options"
    optim_options::Optim.Options = Optim.Options()
    "Observed model - if `nothing`is used, a normal distributed additive error with fixed variance is assumed."
    observed::Union{ObservedModel, Nothing} = nothing
    "AD Backendend"
    ad_backend::AD.AbstractBackend = AD.ForwardDiffBackend()
    "Optimiser"
    optimiser::Optimisers.AbstractRule = ADAM()
end

Base.print(io::IO, ::Reinforce) = print(io, "Reinforce")
Base.summary(io::IO, x::Reinforce) = print(io, x)

function reinforce_loss(candidates, p, alg)
    @unpack loss, reward = alg
    losses = map(loss, candidates)
    rewards = reward(losses)
    # ∇U(θ) = E[∇log(p)*R(t)]
    mean(map(enumerate(candidates)) do (i, candidate)
             rewards[i] * -candidate(p)
         end)
end

function update_parameters!(cache::SearchCache{<:Reinforce})
    @unpack alg, optimiser_state, candidates, keeps, p = cache
    @unpack ad_backend = alg

    ∇p, _... = AD.gradient(ad_backend, (p) -> reinforce_loss(candidates[keeps], p, alg), p)
    opt_state, p_ = Optimisers.update!(optimiser_state, p[:], ∇p[:])
    cache.p .= p_
    return
end
