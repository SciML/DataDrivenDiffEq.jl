"""
$(TYPEDEF)

Common options for all methods provided via `DataDrivenDiffEq`. 

# Fields
$(FIELDS)
    
"""
@with_kw struct DataDrivenCommonOptions{T, K}
    """Maximum iterations"""
    maxiter::Int = 1_00
    """Absolute tolerance"""
    abstol::T = sqrt(eps())
    """Relative tolerance"""
    reltol::T = sqrt(eps())

    """Show a progress"""
    progress::Bool = false
    """Display log - Not implemented right now"""
    verbose::Bool = false

    """Denoise the data using singular value decomposition"""
    denoise::Bool = false
    """Normalize the data"""
    normalize::Bool = false
    #"""Sample options, see `DataSampler`"""
    #sampler::AbstractSampler = 
    #
    #"""Mapping from the candidate solution of a problem to features used for pareto analysis"""
    #f::Function
    #"""Scalarization of the features for a candidate solution"""
    #g::Function
    """Significant digits for the parameters - used for rounding. Default = 10"""
    digits::Int = 10

    """Additional kwargs"""
    kwargs::K = (;)
end
