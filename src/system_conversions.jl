import ModelingToolkit.SymbolicUtils.FnType

function _generate_deqs(x::Basis, states, iv, p)
    @assert length(x) == length(states)  
    # Create new variables with time dependency
    ∂t = Differential(iv)
    dvs = [Num(Sym{FnType{Tuple{Any}, Real}}(_get_name(xi))(value(iv))) for xi in states] 
    dvsdt = ∂t.(dvs)
    # Adapt equations
    eqs = dvsdt .~ x(dvs,p,iv)
    return eqs, dvs
end

## System Conversion
function ModelingToolkit.ODESystem(x::Basis, iv = nothing, dvs = Num[], ps = Num[]; pins = Num[], observed = Num[], systems = ODESystem[],kwargs...)
    iv = isnothing(iv) ? independent_variable(x) : iv
    dvs = isempty(dvs) ? variables(x) : dvs
    ps = isempty(ps) ? parameters(x) : ps
    eqs, dvs = _generate_deqs(x, dvs, iv, ps)
    pins = isempty(pins) ? x.pins : pins
    observed = isempty(observed) ? x.observed : observed
    systems = isempty(systems) ? x.systems : systems
    return ODESystem(
        eqs, iv, dvs, ps,
        pins = pins, observed = observed, systems = systems, kwargs...)
end

function _remove_controls(states, controls)
    idxs = ones(Bool, size(states)...)
    for i in eachindex(states)
        idxs[i] = !any([isequal(states[i], ci) for ci in controls])
    end
    return states[idxs]
end

function _generate_deqs(x::Basis, states, iv, p, controls)
    # Create new variables with time dependency
    states_ = _remove_controls(states, controls)
    @assert length(x) == length(states_)
    ∂t = Differential(iv)
    dvs = [Num(Sym{FnType{Tuple{Any}, Real}}(_get_name(xi))(value(iv))) for xi in states_] 
    input_states = _create_input_vec(states, dvs, controls)
    dvsdt = ∂t.(dvs)
    # Adapt equations
    eqs = dvsdt .~ x(input_states,p, iv)
    return eqs, dvs, input_states
end

function _create_input_vec(states, dvs, controls)
    input_states = Array{Any}(undef, size(states)...)
    state_idx = ones(Bool, size(states)...)
    control_idx = zeros(Bool, size(states))
    for i in eachindex(states)
        for k in eachindex(controls)
            if isequal(states[i], controls[k])
                control_idx[i] = true
                state_idx[i] = false
            end
        end
    end
    input_states[state_idx] .= dvs
    input_states[control_idx] .= controls
    return input_states
end

function ModelingToolkit.ControlSystem(loss, x::Basis, controls, iv = nothing, dvs = nothing, ps = nothing; 
    pins = Num[], observed = Num[], systems = ODESystem[], kwargs...)
    iv = isnothing(iv) ? independent_variable(x) : iv
    dvs = isnothing(dvs) ? variables(x) : dvs
    ps = isnothing(ps) ? parameters(x) : ps
    eqs, dvs, input_states = _generate_deqs(x, dvs, iv, ps, controls)
    #return input_states
    subs = [(xi => is) for (xi, is) in zip(variables(x), input_states)]
    loss = substitute.(loss, (subs,))[1]
    pins = isempty(pins) ? x.pins : pins
    observed = isempty(observed) ? x.observed : observed
    systems = isempty(systems) ? x.systems : systems
    return ControlSystem(loss, eqs, iv, dvs, controls, ps,
        pins = pins, observed = observed, systems = systems, kwargs...)
end


