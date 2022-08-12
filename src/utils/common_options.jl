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
@with_kw struct DataDrivenCommonOptions{T, L, K, PL, PR}
    """Maximum iterations"""
    maxiters::Int = 1_00
    """Absolute tolerance"""
    abstol::T = sqrt(eps())
    """Relative tolerance"""
    reltol::T = sqrt(eps())
    """Show a progress meter"""
    progress::Bool = false
    """Display log - Not implemented right now"""
    verbose::Bool = false
    """Denoise the data using the [`optimal threshold`](https://arxiv.org/abs/1305.5870) method."""
    denoise::Bool = false
    """Normalize the data"""
    normalize::Bool = false
    """Sample options, see [`DataSampler`](@ref)"""
    sampler::AbstractSampler = DataSampler()
    """Significant digits for the parameters - used for rounding. Default = 10"""
    digits::Int = 10
    """Evaluate the expression, see [`Symbolics.build_function`](https://symbolics.juliasymbolics.org/stable/manual/build_function/)"""
    eval_expresssion::Bool = true
    """Linear solve algorithm to use. See [LinearSolve.jl](http://linearsolve.sciml.ai/dev/) for a list of available options."""
    linsolve::L = nothing
    """Left preconditioner to use"""
    Pl::PL = nothing
    """Right preconditioner to use"""
    Pr::PR = nothing
    """Additional kwargs"""
    kwargs::K = (;)
end
