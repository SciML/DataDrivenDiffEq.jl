struct SparseRegressionResult{X <: AbstractArray, L, T, TE} <: AbstractDataDrivenResult
    "Coefficient matrix"
    coefficients::X
    "Number of nonzeros coefficients"
    dof::Int
    "Threshold"
    lambda::L
    "Number of iterations"
    iterations::Int
    """L2 norm error of the testing dataset"""
    testerror::T
    """L2 norm error of the training dataset"""
    trainerror::TE
    """Returncode"""
    retcode::DDReturnCode
end

is_success(k::SparseRegressionResult) = getfield(k, :retcode) == DDReturnCode(1)
l2error(k::SparseRegressionResult) = is_success(k) ? getfield(k, :testerror) : Inf

function l2error(k::SparseRegressionResult{<:Any, <:Any, Nothing})
    is_success(k) ? getfield(k, :traineerror) : Inf
end

get_coefficients(k::SparseRegressionResult) = getfield(k, :coefficients)
