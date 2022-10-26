# This will get called within init in DataDrivenDiffEq

function DataDrivenDiffEq.get_fit_targets(::A, prob::ABSTRACT_CONT_PROB,
                         basis::AbstractBasis) where {A <: AbstractKoopmanAlgorithm}
    @unpack DX, X, p, t, U = prob

    @assert size(DX, 1)==size(X, 1) "$(A) needs equal number of observed states and differentials for continuous problems!"

    Θ = basis(prob)
    jac = jacobian(basis)
    Ỹ = similar(Θ)
    foreach(axes(DX, 2)) do i
        Ỹ[:, i] .= jac(DX[:, i], X[:, i], p, t[i], U[:, i]) * DX[:, i]
    end
    return Θ, Ỹ, DX
end

function DataDrivenDiffEq.get_fit_targets(::A, prob::ABSTRACT_DISCRETE_PROB,
                         basis::AbstractBasis) where {A <: AbstractKoopmanAlgorithm}
    # TODO Maybe we could, but this would require X[:, i+2] -> split in three here
    @assert !is_implicit(basis) "$(A) does not support implicit arguments in the basis for discrete problems!"

    @unpack X, p, t, U = prob
    # Lift 
    Θ = basis(prob)
    n_b, m = size(Θ)
    Ỹ = zeros(eltype(Θ), n_b, m)
    foreach(1:m) do i
        basis(Ỹ[:, i], DataDrivenDiffEq.__EMPTY_VECTOR, X[:, i + 1], p, t[i + 1],
              U[:, i + 1])
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
    results = alg(traindata, testdata, control_idx, options; kwargs...)

    # Get the best result based on test error, if applicable else use testerror
    sort!(results, by = l2error)
    # Convert to basis
    best_res = first(results)
    new_basis = convert_to_basis(best_res, basis, problem, options,control_idx)
    # Build DataDrivenResult
    DataDrivenSolution(
        new_basis,  problem, alg, results, prob, best_res.retcode
    )
end

function convert_to_basis(res::KoopmanResult, basis::Basis, prob, options, control_idx)
    @unpack digits = options
    @unpack C, K, B = res
    control_idx = map(any, eachrow(control_idx))
    # Build the Matrix
    Θ = zeros(eltype(C), size(C, 1), length(basis))
    Θ[:, .! control_idx] .= C*Matrix(K)
    Θ[:, control_idx] .= C*B
    Θ .= round.(Θ, digits = digits)
    DataDrivenDiffEq.__construct_basis(Θ, basis, prob)
end

function (algorithm::AbstractKoopmanAlgorithm)(traindata, testdata, control_idx, options;
                                       control_input = nothing, kwargs...)
    @unpack abstol = options
    # Preprocess control idx, indicates if any control is active in a single basis atom
    control_idx = map(any, eachrow(control_idx))
    no_controls = .!control_idx
    X̄, _, Z̄ = testdata
    map(traindata) do (X, Y, Z)
        K, B = algorithm(X[no_controls, :], Y[no_controls, :], X[control_idx, :], control_input)
        Q = Y[no_controls, :] * X'
        P = X * X'
        C = Z \ Y[no_controls, :]
        trainerror = sum(abs2, Z .- C * (K * X[no_controls, :] .+ B * X[control_idx]))
        if !isempty(X̄)
            testerror = sum(abs2, Z̄ .- C * (K * X̄[no_controls, :] .+ B * X̄[control_idx]))
            retcode = testerror <= abstol ? DDReturnCode(1) : DDReturnCode(5)
        else
            testerror = nothing
            retcode = trainerror <= abstol ? DDReturnCode(1) : DDReturnCode(5)
        end
        KoopmanResult(K, B, C, Q, P, trainerror, testerror, retcode)
    end
end
