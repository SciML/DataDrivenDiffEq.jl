function __assert_linearity(eqs::AbstractVector{Equation}, x::AbstractVector)
    return __assert_linearity(map(x -> Num(x.rhs), eqs), x)
end

# Returns true iff x is not in the arguments of the jacobian of eqs
function __assert_linearity(eqs::AbstractVector{Num}, x::AbstractVector)
    j = Symbolics.jacobian(eqs, x)
    # Check if any of the variables is in the jacobian
    v = get_variables.(j)
    isempty(v) && return true
    v = unique(v)
    for xi in x, vi in v
        isequal(xi, vi) && return false
    end
    return true
end

function assert_lhs(prob::ABSTRACT_CONT_PROB)
    return :continuous, 0.0
end

function assert_lhs(prob::ABSTRACT_DISCRETE_PROB)
    return :discrete, has_timepoints(prob) ? mean(diff(independent_variable(prob))) : 1.0
end

function assert_lhs(prob::AbstractDataDrivenProblem)
    return :direct, 0.0
end

function assert_lhs(prob::DataDrivenDataset)
    return assert_lhs(first(prob.probs))
end

function _generate_variables(sym::Symbol, n::Int, offset::Int = 0)
    xs = [Symbolics.variable(sym, i) for i in (offset + 1):(offset + n)]
    Num.(map(ModelingToolkit.tovar, xs))
end

function _generate_parameters(sym::Symbol, n::Int, offset::Int = 0)
    xs = [Symbolics.variable(sym, i) for i in (offset + 1):(offset + n)]
    Num.(map(ModelingToolkit.toparam, xs))
end

function _set_default_val(x::Num, val::T) where {T <: Number}
    Num(Symbolics.setdefaultval(Symbolics.unwrap(x), val))
end

function __build_eqs(coeff_mat, basis, prob)
    # Create additional variables
    sp = sum(.!iszero.(coeff_mat))
    sps = norm.(eachrow(coeff_mat), 0)
    pl = length(parameters(basis))

    p = _generate_parameters(:p, sp, pl)
    p = collect(p)

    eqs = zeros(Num, size(coeff_mat, 1))
    eqs_ = [e.rhs for e in equations(basis)]
    cnt = 1

    for i in axes(coeff_mat, 1)
        if sps[i] == zero(eltype(coeff_mat))
            continue
        end
        for j in axes(coeff_mat, 2)
            if iszero(coeff_mat[i, j])
                continue
            end
            p[cnt] = _set_default_val(p[cnt], coeff_mat[i, j])
            eqs[i] += p[cnt] * eqs_[j]
            cnt += 1
        end
    end

    return is_implicit(basis) ? _implicit_build_eqs(basis, eqs, p, prob) :
           _explicit_build_eqs(basis, eqs, p, prob)
end

function _explicit_build_eqs(basis, eqs, p, prob)
    causality, dt = assert_lhs(prob)

    xs = states(basis)

    # Else just keep equations, since its a direct problem
    if causality == :continuous
        d = Differential(get_iv(basis))
        eqs = [d(xs[i]) ~ eq for (i, eq) in enumerate(eqs)]
    elseif causality == :discrete
        d = Difference(get_iv(basis), dt = dt)
        eqs = [d(xs[i]) ~ eq for (i, eq) in enumerate(eqs)]
    else
        phi = [Symbolics.variable(Symbol("φ"), i) for i in 1:length(eqs)]
        eqs = [phi[i] ~ eq for (i, eq) in enumerate(eqs)]
    end

    return eqs, Num.(p), Num[]
end

function _implicit_build_eqs(basis, eqs, p, prob)
    implicits = implicit_variables(basis)
    if __assert_linearity(eqs, implicits)
        eqs = eqs .~ 0
        try
            # Try to solve the eq for the implicits
            eqs = ModelingToolkit.solve_for(eqs, implicits)
            eqs = implicits .~ eqs
            implicits = Num[]
        catch
            @warn "Failed to solve recovered equations for implicit variables. Returning implicit equations."
        end
    end

    return eqs, Num.(p), implicits
end

function __construct_basis(X, b, prob, options)
    @unpack eval_expresssion, generate_symbolic_parameters = options
    @unpack p = prob
    
    if generate_symbolic_parameters
        eqs, ps, implicits = __build_eqs(X, b, prob)

        p_ = parameters(b)

        pss = map(eachindex(p)) do i
            _set_default_val(Num(p_[i]), p[i])
        end
        p_new = [pss; ps]
    else
        eqs = X*reduce(vcat, map(x->x.rhs, equations(b)))
        ps = parameters(b)
        eqs, ps, implicits = is_implicit(b) ? _implicit_build_eqs(b, eqs, ps, prob) : _explicit_build_eqs(b, eqs, ps, prob)
        
        p_new = map(eachindex(p)) do i
            _set_default_val(Num(ps[i]), p[i])
        end
    end
        
    Basis(eqs, states(b),
          parameters = p_new, iv = get_iv(b),
          controls = controls(b), observed = observed(b),
          implicits = implicits,
          name = gensym(:Basis),
          eval_expression = eval_expresssion)
end

function unit_basis(prob::DataDrivenProblem)
    @unpack X, p, t, U, Y, DX = prob
    n_x = size(X, 1)
    n_p = size(p, 1)
    n_u = size(U, 1)

    t = Num(ModelingToolkit.tovar(Symbolics.variable(:t)))
    x = _generate_variables(:x, n_x)
    p = _generate_parameters(:p, n_p)
    u = _generate_variables(:u, n_u)

    Basis([x; u], x, controls = u, independent_variable = t, parameters = p)
end
