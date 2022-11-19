struct KoopmanResult{K, B, C, Q, P, T, TE} <: AbstractDataDrivenResult
    """Matrix representation of the operator / generator"""
    k::K
    """Matrix representation of the inputs mapping"""
    b::B
    """Matrix representation of the pullback onto the states"""
    c::C
    """Internal matrix used for updating"""
    q::Q
    """Internal matrix used for updating"""
    p::P
    """L2 norm error of the testing dataset"""
    testerror::T
    """L2 norm error of the training dataset"""
    trainerror::TE
    """Returncode"""
    retcode::DDReturnCode
end

is_success(k::KoopmanResult) = getfield(k, :retcode) == DDReturnCode(1)
l2error(k::KoopmanResult) = is_success(k) ? getfield(k, :testerror) : Inf

function l2error(k::KoopmanResult{<:Any, <:Any, <:Any, <:Any, <:Any, Nothing})
    is_success(k) ? getfield(k, :traineerror) : Inf
end

get_operator(k::KoopmanResult) = getfield(k, :k)
get_generator(k::KoopmanResult) = getfield(k, :k)

get_inputmap(k::KoopmanResult) = getfield(k, :b)
get_outputmap(k::KoopmanResult) = getfield(k, :c)

get_trainerror(k::KoopmanResult) = getfield(k, :trainerror)
get_testerror(k::KoopmanResult) = getfield(k, :testerror)
