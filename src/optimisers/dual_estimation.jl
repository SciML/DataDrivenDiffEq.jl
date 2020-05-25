mutable struct DualOptimiser{B, S, O, P, R} <: AbstractOptimiser
    basis::B # basis needed for iip update
    param_opt::O # Optim optimiser
    sparse_opt::S # Sparse optimiser
    ps::P # parameters
    bounds::R # bounds
end

"""
    DualOptimiser(basis, optim_opt, sparse_opt)

`DualOptimiser` allows the sparse regression for a parametrized `Basis`.
"""
function DualOptimiser(basis, opt, sparse_opt)
    p = randn(length(parameters(basis)))
    bounds = ([Inf for i in 1:length(p)], [-Inf for i in 1:length(p)])
    return DualOptimiser(basis, opt, sparse_opt, p, bounds)
end

function set_threshold!(opt::DualOptimiser, threshold)
    set_threshold!(opt.sparse_opt, threshold)
end

get_threshold(opt::DualOptimiser) = get_threshold(opt.sparse_opt)

init(opt::DualOptimiser, A::AbstractArray, Y::AbstractArray) = init(opt.sparse_opt, A, Y)
init!(X::AbstractArray, opt::DualOptimiser, A::AbstractArray, Y::AbstractArray) = init!(X, opt.sparse_opt, A, Y)

struct Evaluator{F, G, H}
    f::F
    g::G
    h::H
end

function (c::Evaluator)(Ξ, X , DX, t)
	# Create closure
    function f(p)
		cost = zero(eltype(X))
        @inbounds for i in 1:size(X, 2)
            cost += c.f(p, [[Ξ...]; X[:, i]; DX[:, i]; t[i]])
		end
		return cost
	end

	function g!(g, p)
        g .= zero(g)
		@inbounds for i in 1:size(X, 2)
			g .+= c.g(p, [[Ξ...]; X[:, i]; DX[:, i]; t[i]])
		end
		return nothing
	end

	function h!(h, p)
        h .= zero(h)
		@inbounds for i in 1:size(X, 2)
			h .+= c.h(p, [[Ξ...]; X[:, i]; DX[:, i]; t[i]])
		end
		return nothing
	end

    function fgh!(F, G, H, x)
        G == nothing || g!(G, x)
        H == nothing || h!(H, x)
        F == nothing || return f(x)
        nothing
    end

    return Optim.only_fgh!(fgh!)
end
