## Helper for the solution

# Check linearity

function assert_linearity(eqs::AbstractVector{Equation}, x::AbstractVector)
    return assert_linearity(map(x->Num(x.rhs), eqs), x)
end

# Returns true iff x is not in the arguments of the jacobian of eqs
function assert_linearity(eqs::AbstractVector{Num}, x::AbstractVector)
    j = Symbolics.jacobian(eqs, x)
    # Check if any of the variables is in the jacobian
    v = unique(reduce(vcat, map(get_variables, j)))
    for xi in x, vi in v
        isequal(xi, vi) && return false
    end
    return true
end

function construct_basis(X, b, implicits = Num[]; dt = one(eltype(X)), lhs::Symbol = :continuous, is_implicit = false, eval_expression = false)

    # Create additional variables
    sp = sum(.! iszero.(X))
    sps = norm.(eachcol(X), 0)
    inds = sps .> zero(eltype(X))
    pl = length(parameters(b))
    
    p = [Symbolics.variable(:p, i) for i in (pl+1):(pl+sp)]
    p = collect(p)
    ps = zeros(eltype(X), sp)

    eqs = zeros(Num, sum(inds))
    eqs_ = [e.rhs for e in equations(b)]
    cnt = 1
    for j in 1:size(X, 2)
        if sps[j] == zero(eltype(X))
            continue
        end
        for i in 1:size(X, 1)
            if iszero(X[i,j])
                continue
            end
            ps[cnt] = X[i,j]
            eqs[j] += p[cnt]*eqs_[i]
            cnt += 1
        end
    end

    # Build the lhs
    xs = states(b)
    if isempty(implicits) || !is_implicit
        if length(eqs) == length(states(b))
            if lhs == :continuous
                d = Differential(get_iv(b))
                eqs = [d(xs[i]) ~ eq for (i,eq) in enumerate(eqs)]
            elseif lhs == :discrete
                d = Difference(get_iv(b), dt = dt)
                eqs = [d(xs[i]) ~ eq for (i,eq) in enumerate(eqs)]
            end
        end
    else
        if !isempty(implicits)
            eqs = eqs .~ 0
            if assert_linearity(eqs, implicits)
                try
                    # Try to solve the eq for the implicits
                    eqs = ModelingToolkit.solve_for(eqs, implicits)
                    eqs = implicits .~ eqs
                    implicits = []
                catch 
                    @warn "Failed to solve recovered equations for implicit variables. Returning implicit equations."
                end
            end
        end
    end

    Basis(
        eqs, xs,
        parameters = [parameters(b); p], iv = get_iv(b),
        controls = controls(b), observed = observed(b),
        implicits = implicits,
        name = gensym(:Basis),
        eval_expression = eval_expression
    ), ps
end

function assert_lhs(prob::AbstracContProb)
    return :continuous, 0.0
end

function assert_lhs(prob::AbstractDiscreteProb)
    return :discrete, mean(independent_variable(prob))
end

function assert_lhs(prob::AbstractDataDrivenProblem)
    return :direct, 0.0
end

function assert_lhs(prob::DataDrivenDataset)
    return assert_lhs(first(prob.probs))
end

