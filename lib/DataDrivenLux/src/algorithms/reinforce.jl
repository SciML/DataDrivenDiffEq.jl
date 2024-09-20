@concrete struct Reinforce <: AbstractDAGSRAlgorithm
    reward
    ad_backend <: AD.AbstractBackend
    options <: CommonAlgOptions
end

"""
$(SIGNATURES)

Uses the REINFORCE algorithm to search over the space of possible solutions to the
symbolic regression problem.
"""
function Reinforce(; reward = RelativeReward(false), populationsize = 100,
        functions = (sin, exp, cos, log, +, -, /, *), arities = (1, 1, 1, 1, 2, 2, 2, 2),
        n_layers = 1, skip = true, loss = aicc, keep = 0.1, use_protected = true,
        distributed = false, threaded = false, rng = Random.default_rng(),
        optimizer = LBFGS(), optim_options = Optim.Options(), observed = nothing,
        alpha = 0.999f0, optimiser = Adam(), ad_backend = AD.ForwardDiffBackend())
    return Reinforce(reward,
        ad_backend,
        CommonAlgOptions(;
            populationsize, functions, arities, n_layers, skip, simplex = Softmax(), loss,
            keep, use_protected, distributed, threaded, rng, optimizer,
            optim_options, optimiser, observed, alpha))
end

Base.print(io::IO, ::Reinforce) = print(io, "Reinforce")
Base.summary(io::IO, x::Reinforce) = print(io, x)

function reinforce_loss(candidates, p, alg)
    losses = map(alg.options.loss, candidates)
    rewards = alg.reward(losses)
    # ∇U(θ) = E[∇log(p)*R(t)]
    return mean(map(enumerate(candidates)) do (i, candidate)
        return rewards[i] * -candidate(p)
    end)
end

function update_parameters!(cache::SearchCache{<:Reinforce})
    (; alg, optimiser_state, candidates, keeps, p) = cache
    (; ad_backend) = alg

    ∇p, _... = AD.gradient(ad_backend, (p) -> reinforce_loss(candidates[keeps], p, alg), p)
    opt_state, p_ = Optimisers.update!(optimiser_state, p[:], ∇p[:])
    cache.p .= p_
    return
end
