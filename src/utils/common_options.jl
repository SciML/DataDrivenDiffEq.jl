"""
$(TYPEDEF)

Common options for all methods provided via `DataDrivenDiffEq`. 

# Fields
$(FIELDS)
    
## Note

The keyword argument `eval_expression` controls the function creation
behavior. `eval_expression=true` means that `eval` is used, so normal
world-age behavior applies (i.e. the functions cannot be called from
the function that generates them). If `eval_expression=false`,
then construction via GeneralizedGenerated.jl is utilized to allow for
same world-age evaluation. However, this can cause Julia to segfault
on sufficiently large basis functions. By default eval_expression=false.
"""
@with_kw struct DataDrivenCommonOptions{T, K}
    """Maximum iterations"""
    maxiter::Int = 1_00
    """Absolute tolerance"""
    abstol::T = sqrt(eps())
    """Relative tolerance"""
    reltol::T = sqrt(eps())
    """Show a progress meter"""
    progress::Bool = false
    """Display log - Not implemented right now"""
    verbose::Bool = false
    """Denoise the data using the [`optimal threshold`](@ref optimal_shrinkage) method."""
    denoise::Bool = false
    """Normalize the data"""
    normalize::Bool = false
    """Sample options, see [`DataSampler`](@ref)"""
    sampler::AbstractSampler = DataSampler()
    """Significant digits for the parameters - used for rounding. Default = 10"""
    digits::Int = 10
    """Evaluate the expression, see [`Symbolics.build_function`](@ref)"""
    eval_expresssion::Bool = true
    """Additional kwargs"""
    kwargs::K = (;)
end
