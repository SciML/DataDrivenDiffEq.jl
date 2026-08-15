"""
$(TYPEDEF)

`WyNDA` implements the Wide-Array of Nonlinear Dynamics Approximation update as an
online recursive least-squares estimator with exponential forgetting. Given a
matrix of evaluated basis functions `X` and targets `Y`, it updates the
coefficient matrix sample-by-sample so that `Y ≈ coefficients * X`.

The forgetting factor `λ` controls how quickly older samples are discounted.
Values close to one recover a batch least-squares-like fit, while smaller values
adapt faster to parameter drift.

# Arguments

- `λ`: forgetting factor satisfying `0 < λ <= 1`.

# Keywords

- `initial_covariance`: positive scalar or square matrix used for the initial
  inverse covariance.
- `initial_coefficients`: optional initial coefficient vector or target-by-feature
  matrix.

# Returns

Return an online algorithm object callable as `alg(X, Y; options)`, producing the
coefficient matrix, the forgetting factor, and the number of observations.

# Fields

$(FIELDS)

# Example

```julia
opt = WyNDA()
opt = WyNDA(0.998)
opt = WyNDA(0.998; initial_covariance = 1.0e4)
```
"""
struct WyNDA{T <: Number, C, IC} <: AbstractSparseRegressionAlgorithm
    """Exponential forgetting factor."""
    λ::T
    """Initial inverse covariance scale or matrix."""
    initial_covariance::C
    """Optional initial coefficient vector or matrix."""
    initial_coefficients::IC

    function WyNDA(
            λ::T = 1.0;
            initial_covariance::C = 1.0e6,
            initial_coefficients::IC = nothing
        ) where {T <: Number, C, IC}
        @assert zero(T) < λ <= one(T) "Forgetting factor λ must satisfy 0 < λ <= 1"
        if initial_covariance isa Number
            @assert initial_covariance > zero(initial_covariance) "Initial covariance must be positive"
        end
        return new{T, C, IC}(λ, initial_covariance, initial_coefficients)
    end
end

Base.summary(::WyNDA) = "WyNDA"

function _initial_covariance(alg::WyNDA, ::Type{T}, n_features::Int) where {T}
    covariance = alg.initial_covariance
    if covariance isa Number
        return Matrix{T}(I, n_features, n_features) .* T(covariance)
    end
    @assert size(covariance) == (n_features, n_features) "Initial covariance matrix size must match the number of features"
    return Matrix{T}(covariance)
end

function _initial_coefficients(
        alg::WyNDA, ::Type{T}, n_targets::Int, n_features::Int
    ) where {T}
    coefficients = alg.initial_coefficients
    if coefficients === nothing
        return zeros(T, n_targets, n_features)
    end
    if coefficients isa AbstractVector
        @assert n_targets == 1 "Initial coefficient vectors are only valid for single-target problems"
        @assert length(coefficients) == n_features "Initial coefficient vector length must match the number of features"
        return reshape(T.(coefficients), 1, n_features)
    end
    @assert size(coefficients) == (n_targets, n_features) "Initial coefficient matrix size must match targets by features"
    return Matrix{T}(coefficients)
end

function (alg::WyNDA)(
        X::AbstractMatrix, Y::AbstractVecOrMat;
        options::DataDrivenCommonOptions = DataDrivenCommonOptions(),
        kwargs...
    )
    Y_matrix = Y isa AbstractVector ? reshape(Y, 1, :) : Y
    @assert size(X, 2) == size(Y_matrix, 2) "X and Y must have the same number of observations"

    n_features, n_observations = size(X)
    n_targets = size(Y_matrix, 1)
    T = float(promote_type(eltype(X), eltype(Y_matrix), typeof(alg.λ)))

    coefficients = _initial_coefficients(alg, T, n_targets, n_features)
    covariance = _initial_covariance(alg, T, n_features)
    λ = T(alg.λ)

    for k in 1:n_observations
        basis_values = view(X, :, k)
        targets = view(Y_matrix, :, k)
        covariance_basis = covariance * basis_values
        denominator = λ + dot(basis_values, covariance_basis)
        gain = covariance_basis ./ denominator
        residual = targets .- coefficients * basis_values
        coefficients .+= residual * adjoint(gain)
        covariance .-= gain * adjoint(covariance_basis)
        covariance ./= λ
    end

    return coefficients, alg.λ, n_observations
end
