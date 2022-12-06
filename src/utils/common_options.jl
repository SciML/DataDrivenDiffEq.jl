

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

Denoising happens before normalization!
"""
@with_kw struct DataDrivenCommonOptions{T, K}
    # Optimization options
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
    # Preprocessing
    """Denoise the data using the [`optimal threshold`](https://arxiv.org/abs/1305.5870) method."""
    denoise::Bool = false
    """Normalize the data, see `DataNormalization`"""
    normalize::DataNormalization = DataNormalization()
    """Data processing pipeline, see `DataProcessing`"""
    data_processing::DataProcessing = DataProcessing()
    # Postprocessing
    """Rounding mode for the parameters"""
    roundingmode::RoundingMode = RoundToZero
    """Digits for the parameters - used for rounding."""
    digits::Int = 10
    #"""Significant digits for the parameters - used for rounding."""
    #sigdigits::Int = 10
    """Enables the use of symbolic parameters for the result. If `false`, the numerical value is used."""
    generate_symbolic_parameters::Bool = true
    """Evaluate the expression, see [`Symbolics.build_function`](https://symbolics.juliasymbolics.org/stable/manual/build_function/)"""
    eval_expresssion::Bool = false
    """Additional kwargs"""
    kwargs::K = (;)
end
