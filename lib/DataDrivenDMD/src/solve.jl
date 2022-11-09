# This will get called within init in DataDrivenDiffEq

function DataDrivenDiffEq.get_fit_targets(::A, prob::ABSTRACT_CONT_PROB,
                                          basis::AbstractBasis) where {
                                                                       A <:
                                                                       AbstractKoopmanAlgorithm
                                                                       }
    @unpack DX, X, p, t, U = prob

    @assert size(DX, 1)==size(X, 1) "$(A) needs equal number of observed states and differentials for continuous problems!"

    Θ = basis(prob)
    n_t = size(Θ, 1)
    n_x = size(X, 1)
    jac = let n_t = n_t, n_x = n_x, f = jacobian(basis)
        (args...) -> reshape(f(args...), n_x, n_t)
    end
    Ỹ = similar(Θ)
    if is_controlled(basis)
        foreach(axes(DX, 2)) do i
            Ỹ[:, i] .= jac(X[:, i], p, t[i], U[:, i]) * DX[:, i]
        end
    else
        foreach(axes(DX, 2)) do i
            Ỹ[:, i] .= jac(X[:, i], p, t[i]) * DX[:, i]
        end
    end
    return Θ, Ỹ, DX
end

function DataDrivenDiffEq.get_fit_targets(::A, prob::ABSTRACT_DISCRETE_PROB,
                                          basis::AbstractBasis) where {
                                                                       A <:
                                                                       AbstractKoopmanAlgorithm
                                                                       }
    # TODO Maybe we could, but this would require X[:, i+2] -> split in three here
    @assert !is_implicit(basis) "$(A) does not support implicit arguments in the basis for discrete problems!"

    @unpack X, p, t, U = prob
    # Lift 
    Θ = basis(prob)
    n_b, m = size(Θ)
    Ỹ = zeros(eltype(Θ), n_b, m)

    if is_controlled(basis)
        foreach(1:m) do i
            Ỹ[:, i] .= basis(X[:, i + 1], p, t[i + 1],
                               U[:, i + 1])
        end
    else
        foreach(1:m) do i
            Ỹ[:, i] .= basis(X[:, i + 1], p, t[i + 1])
        end
    end
    return Θ, Ỹ, X[:, 2:end]
end

## Solve the Koopman 
function CommonSolve.solve!(prob::InternalDataDrivenProblem{A}) where {
                                                                       A <:
                                                                       AbstractKoopmanAlgorithm
                                                                       }
    @unpack alg, basis, testdata, traindata, control_idx, options, problem, kwargs = prob

    # Check for 
    results = alg(prob; kwargs...)

    # Get the best result based on test error, if applicable else use testerror
    sort!(results, by = l2error)
    # Convert to basis
    best_res = first(results)
    new_basis = convert_to_basis(best_res, basis, problem, options, control_idx)
    # Build DataDrivenResult
    DataDrivenSolution(new_basis, problem, alg, results, prob, best_res.retcode)
end

function convert_to_basis(res::KoopmanResult, basis::Basis, prob, options, control_idx)
    @unpack c, k, b = res
    control_idx = map(any, eachrow(control_idx))
    # Build the Matrix
    Θ = zeros(eltype(c), size(c, 1), length(basis))
    
    if any(control_idx)
        Θ[:, .!control_idx] .= c * Matrix(k)
        Θ[:, control_idx] .= c * b
    else
        Θ .= c * Matrix(k)
    end
    
    DataDrivenDiffEq.__construct_basis(Θ, basis, prob, options)
end

function __compute_rss(Z, C, K, B, X, U)
    begin
        (isempty(U) || isempty(B)) && return sum(abs2, Z .- C * (K * X))
        return sum(abs2, Z .- C * (K * X + B * U))
    end
end

function (algorithm::AbstractKoopmanAlgorithm)(prob::InternalDataDrivenProblem;
                                               control_input = nothing, kwargs...)
    @unpack traindata, testdata, control_idx, options = prob
    @unpack abstol = options
    # Preprocess control idx, indicates if any control is active in a single basis atom
    control_idx = map(any, eachrow(control_idx))
    no_controls = .!control_idx

    X̃, _, Z̃ = testdata

    if any(control_idx) && !isempty(X̃)
        X̃, Ũ = X̃[no_controls, :], X̃[control_idx, :]
    else
        X̃, Ũ = X̃, DataDrivenDiffEq.__EMPTY_MATRIX
    end

    map(traindata) do (X, Y, Z)
        if any(control_idx)
            @info control_idx
            X_, Y_, U_ = X[no_controls, :], Y[no_controls, :], X[control_idx, :]
        else
            X_, Y_, U_ = X, Y, DataDrivenDiffEq.__EMPTY_MATRIX
        end

        K, B = algorithm(X_, Y_, U_, control_input)
        Q = Y_ * X'
        P = X * X'
        C = Z / Y_
        trainerror = __compute_rss(Z, C, K, B, X_, U_)
        if !isempty(X̃)
            testerror = __compute_rss(Z̃, C, K, B, X̃, Ũ)
            retcode = testerror <= abstol ? DDReturnCode(1) : DDReturnCode(5)
        else
            testerror = nothing
            retcode = trainerror <= abstol ? DDReturnCode(1) : DDReturnCode(5)
        end
        KoopmanResult(K, B, C, Q, P, testerror, trainerror, retcode)
    end
end
