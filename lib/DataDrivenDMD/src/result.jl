struct KoopmanResult{K, B, C, Q, P} <: AbstractDataDrivenResult
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
    """Returncode"""
    retcode::Symbol
end
