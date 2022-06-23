## Problem
struct SparseIdentificationProblem{X,PR,B,TR,TS,P,O}
    Ξ::X
    prob::PR
    basis::B
    train::TR
    test::TS
    optimizer::P
    options::O
    eval_expression::Bool
end

## Solution
struct SparseLinearSolution{X,L,S,E,F,O,P}
    Ξ::X
    λ::L
    sets::S
    error::E
    folds::F
    opt::O
    options::P
end

# Selection

select_by(::Val, sol::SparseLinearSolution) = begin
    @unpack Ξ, error, λ = sol
    i = argmin(error)
    return Ξ[i, :, :], error[i], λ[i, :]
end

select_by(::Val{:kfold}, sol::SparseLinearSolution) = begin
    @unpack Ξ, folds, error, λ = sol
    size(Ξ, 1) <= 1 && return select_by(1, sol)
    i = argmin(mean(folds, dims = 1)[1, :])
    return Ξ[i, :, :], error[i], λ[i, :]
end

select_by(::Val{:stat}, sol::SparseLinearSolution) = begin
    @unpack Ξ, folds, error, λ = sol
    size(Ξ, 1) <= 1 && return select_by(1, sol)
    best = argmin(error)
    ξ = mean(Ξ, dims = 1)[1, :, :]
    s = std(Ξ, dims = 1)[1, :, :]
    return measurement.(ξ, s), error[best], λ[best, :]
end

## Solve!

function CommonSolve.init(
    prob::AbstractDataDrivenProblem{N,C,P},
    basis::AbstractBasis,
    opt::AbstractOptimizer,
    args...;
    eval_expression = false,
    kwargs...,
)::SparseIdentificationProblem where {N,C,P}

    @is_applicable prob

    options = DataDrivenCommonOptions(opt; kwargs...)

    @unpack sampler = options

    train, test = sampler(prob)

    Y = get_target(prob)

    n_y = size(Y, 1)

    Ξ = zeros(N, length(train), length(basis), n_y)

    return SparseIdentificationProblem(
        Ξ,
        prob,
        basis,
        train,
        test,
        opt,
        options,
        eval_expression,
    )
end

function CommonSolve.solve!(p::SparseIdentificationProblem)#::DataDrivenSolution

    @unpack Ξ, prob, basis, train, test, optimizer, options, eval_expression = p
    @unpack normalize,
    denoise,
    sampler,
    maxiter,
    abstol,
    reltol,
    verbose,
    progress,
    f,
    g,
    kwargs = options


    T = eltype(Ξ)

    is_implicit = isa(optimizer, AbstractSubspaceOptimizer)

    DX = get_target(prob)

    Θ = zeros(T, length(basis), length(prob))

    @views if is_implicit
        basis(Θ, get_implicit_oop_args(prob)...)
    else
        basis(Θ, prob)
    end

    c = candidate_matrix(basis, size(DX, 1))

    scales = ones(T, length(basis))

    normalize ? normalize_theta!(scales, Θ) : nothing

    denoise ? optimal_shrinkage!(Θ') : nothing

    testerror = zeros(T, size(Ξ, 1))
    trainerror = zeros(T, size(Ξ, 1), size(Ξ, 1))
    λs = zeros(T, size(Ξ, 1), size(DX, 1))

    fg = (x...) -> g(f(x...))


    Aₜ = Θ[:, test]'
    Yₜ = DX[:, test]'

    @views for (i, t) in enumerate(train)
        for (j, cj) in enumerate(eachrow(c))
            A = Θ[cj, t]'
            Y = DX[j:j, t]'
            X = Ξ[i, cj, j:j]

            λs[i, j:j] .= sparse_regression!(
                X,
                A,
                Y,
                optimizer;
                maxiter = maxiter,
                abstol = abstol,
                f = f,
                g = g,
                progress = progress,
                kwargs...,
            )
        end

        for (j, tt) in enumerate(train)
            trainerror[i, j] = fg(Ξ[i, :, :], Θ[:, tt]', DX[:, tt]')
        end

        testerror[i] = fg(Ξ[i, :, :], Aₜ, Yₜ)

        rescale_xi!(Ξ[i, :, :], scales, true)
    end

    sol = SparseLinearSolution(
        Ξ,
        λs,
        (train, test),
        testerror,
        trainerror,
        optimizer,
        options,
    )
    return DataDrivenSolution(
        prob,
        sol,
        basis,
        optimizer,
        implicit_variables(basis);
        eval_expression = eval_expression,
        kwargs...,
    )
end
