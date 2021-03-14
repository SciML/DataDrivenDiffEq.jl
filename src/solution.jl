
struct DataDrivenSolution{R <: AbstractBasis, P, A, O,M} <: AbstractDataDrivenSolution
    """Result"""
    res::R # The result
    """Problem"""
    prob::P
    """Algorithm used for solution"""
    alg::A # Solution algorithm
    """Status of the solution"""
    retcode::Symbol
    """Original ouput"""
    original::O
    """Error metrics"""
    metrics::M
end

function Base.show(io::IO, s::AbstractDataDrivenSolution)
    println(io, string("Retcode: ", s.retcode))
    return
end

#function _build_discrete_solution(basis, prob, alg, retcode, o)
#    # Apply the error metrics
#    return DataDrivenSolution(basis, prob, alg, retcode, o, metrics)
#end
#
#function _build_continuous_solution(basis, prob, alg, retcode, o)
#    # Apply the error metrics
#    return DataDrivenSolution(basis, prob, alg, retcode, o, metrics)
#end
