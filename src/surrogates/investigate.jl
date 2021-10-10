## Struct for all options
struct InvestigationOptions{T, S, O, K}
    """Absolute error tolerance"""
    abstol::T 
    """Relative error tolerance"""
    reltol::T
    """Factor to use within investigation of symmetries"""
    alpha::T
    """Symmetry combinations"""
    sym_ops::S
    """Separation operations"""
    sep_ops::O
    """Additional kwargs"""
    kwargs::K

    function InvestigationOptions(y, f, x)
        T = eltype(y)

        e = max(norm(f(x) - y), 10*eps())

        abstol = convert(T, 10)*e
        reltol = abstol / norm(y)

        alpha = convert(T, 1.2)

        sym_ops = Dict(
            [
                (-, (+, +)),
                (+, (+, -)),
                (*, (*, /)), 
                (/, (*, *))
            ]
        )

        sep_ops = Dict(
            [
                (+, -),
                (-, +), 
                (*, /),
                (/, *),
            ]
        )

        return new{T, typeof(sym_ops), typeof(sep_ops), Dict}(
            abstol, reltol, alpha, sym_ops, sep_ops, Dict(("max_depth" => 3), )
        )
    end
end


## Create a new function

function _create_f(f, args...)
    _f(x::AbstractVector) = f(x)
    _f(x::AbstractMatrix) = hcat(map(_f, eachcol(x))...)
    return _f
end
   
function _create_f(f, i::Int, args...)
    _f(x::AbstractVector) = getindex(f(x), i)
    _f(x::AbstractMatrix) = hcat(map(_f, eachcol(x))...)
    return _f
end

function _set_constant_input(f, ts...)
    function _f(x)
        _x = similar(x)
        _x .= x
        for t in ts
            _x[first(t)] = last(t)
        end
        return f(_x)     
    end
    _create_f(_f)
end

function _composition_f(g, h, op, args...)
    function gh(x)
        broadcast(op, g(x), h(x))
    end
    _create_f(gh)
end

## Investigation options
function create_incidence(f, x::AbstractMatrix{T}, opts::InvestigationOptions, jac = _gradient(f)) where T 
    inc = jac(x[:, 1])
    _inc = similar(inc)
    inc .= zero(T)
    _ , m = size(x)

    @views for xi in eachcol(x)
        _inc .= abs.(jac(xi))
        inc .+= max.(_inc .>= opts.abstol, _inc ./ maximum(_inc) .>= opts.reltol) 
    end

    inc ./= m
    round.(inc)
end

function is_linear(f, x, opts::InvestigationOptions, jac = _gradient(f), args...; kwargs...)
    e = var(map(jac, eachcol(x))) 
    return all((e .<= opts.abstol))
end


function explore_symmetries(f, x, inz, opts::InvestigationOptions, args...; kwargs...)
    ops = opts.sym_ops

    syms = AbstractCoordinateTransform[]
    for (a, b) in ops
        explore_symmetries!(syms, f, x, a, b, inz, opts)
    end
    #unique!(syms)
    syms
end

function explore_symmetries!(s, f, x, a, b, inz, opts::InvestigationOptions, args...; kwargs...)
    
    x̂ = similar(x)
    y = f(x)
    e = zero(eltype(y))

    # Store the inzidenz already found
    founds = Int[]
    for i in 1:size(x, 1)-1
        # Skip if already visited or not part of the incidenz
        i ∈ founds && continue
        inz[i] < 1 && continue
        
        for j in (i+1):size(x,1)
            inz[j] < 1 && continue
            
            x̂ .= x
            x̂[i,:] .= broadcast(first(b), x[i,:], opts.alpha)
            x̂[j,:] .= broadcast(last(b), x[j,:], opts.alpha)

            e = norm(f(x̂) - y)
            
            if (e < opts.abstol) || (e / norm(y) < opts.reltol)
                push!(s, CoordinateTransform(a, [i, j]))
                push!(founds, i, j)
                break
            end
        end
    end
    
    return
end

function separate_function(f, x, inz, opts::InvestigationOptions, args...; depth = 1, kwargs...)
    depth >= opts.kwargs["max_depth"] && return nothing
        
    surrogates = AbstractSurrogate[]
    for (a, b) in opts.sep_ops
        separate_function!(surrogates, f, x, a, b, inz, opts, args...; kwargs...)
    end
    if isempty(surrogates)
        return nothing
    end
    return first(surrogates)
end

function separate_function!(s, f, x, comp, op, inz, opts::InvestigationOptions, args...; depth = 1, kwargs...)
    depth >= opts.kwargs["max_depth"] && return nothing
    # Assume f = comp(g, h)
    # Check via op(f(x), comp(g,h))
    # Returns the first instance to find
    e = zero(eltype(x))
    y = f(x)

    
    # Store the inzidenz already found
    founds = Int[]

    for i in 1:size(x, 1)-1
        # Skip if already visited or not part of the incidenz
        i ∈ founds && continue
        inz[i] < 1 && continue
        
        for j in (i+1):size(x,1)
            inz[j] < 1 && continue
            
            g = _set_constant_input(f, (i, mean(x[i,:])))
            h = _composition_f(
                    _set_constant_input(f, (j, mean(x[j,:]))),
                    _set_constant_input(f, (i, mean(x[i,:])), (j, mean(x[j,:]))),
                    op
            )

            f̂ = _composition_f(g, h, comp)

            e = norm(f(x) - f̂(x))
            
            if (e < opts.abstol) || (e / norm(y) < opts.reltol)
                push!(s, CompositeSurrogate(comp, 
                    explore_surrogate(g, x, opts, depth = depth+1, args...; kwargs...),
                    explore_surrogate(h, x, opts, depth = depth+1, args...; kwargs...)
                ))
                push!(founds, i, j)
                break
            end
        end
    end

    return
    
end

function explore_surrogate(f, x, opts::InvestigationOptions, depth = 0, args...; kwargs...)
    depth >= opts.kwargs["max_depth"] && return nothing
    
    # Check function output
    y = f(x[:,1])
    if size(y, 1) > 1
        _s = map(1:size(y, 1)) do i
            _f = _create_f(f, i)
            explore_surrogate(_f, x, opts, args...; kwargs...)
        end
        filter!(x->!isnothing(x), _s)
        return _s
    end

    # We can check the data domains and supported transformations here
    # Something like exp, log, inv

    # Create inzidenz
    inz = create_incidence(f, x, opts)
    # Check linearity
    f_linear = is_linear(f, x, opts, args...; kwargs...)
    # Check symmetries
    syms = explore_symmetries(f, x, inz, opts)
    # Check special cases 
    ops = get_operator.(syms)
    # Division cancels out linearity check
    #if f_linear && ( (/) ∈ ops)
    #    f_linear = false
    #end

    if f_linear
        coeff = zeros(1, size(x, 1))
        coeff[:, BitVector(inz)] .= f(x) / x[BitVector(inz), :]
        coeff[abs.(coeff) .<= eps()] .= zero(eltype(x))
        return LinearSurrogate(coeff[1,:], inz, syms)
    end

    _s = separate_function(f, x, inz, opts, depth = depth+1)
    !isnothing(_s) && return _s
    
    
    return NonlinearSurrogate(f, inz, transforms = syms)
end
