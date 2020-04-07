# ADD FRACTAL DIMENSION ESTIMATOR OPTIONS
# DEFINE METRICS OF PERFORMANCE LIKE CV
# ADD PARAMETRIC SWEEP ROUTINES
# Research and compare other solvers besides ControlSystems

@with_kw mutable struct HAVOKsim{R}
    sol = [0.0]
    fulltspan::Union{StepRangeLen{R,Base.TwicePrecision{R},Base.TwicePrecision{R}},StepRange{Int,Int}} = 0:0.1:1
    x0::Array{R,1} = [0.0]
    u::Array{R,1} = [0.0]
end


@with_kw mutable struct HAVOKmodel{R}
    timeseries::Array{R,1}
    d::R = 0.0
    q::Int = 0
    r::Int = 0
    dt::R = 1.0
    Embedding::DelayEmbedding = DelayEmbedding()
    NumericalDerivative::Array{R,2} = [0.0 0.0]
    RegressionCoefficient::Array{R,2} = [0.0 0.0]
    sys::StateSpace{R,Array{R,2}} = ss([0.0],[0.0],[0.0],[0.0])
    sim::HAVOKsim = HAVOKsim()
end


timeseries(H::HAVOKmodel) = H.timeseries
delay(H::HAVOKmodel) = H.q
rank(H::HAVOKmodel) = H.r
timestep(H::HAVOKmodel) = H.dt
embedding(H::HAVOKmodel) = H.Embedding
modes(E::DelayEmbedding) = E.Eigenmodes
modes(H::HAVOKmodel; plotQ=false, vars=1:rank(H)) = return (plotQ ? modes_plot(H, vars=vars) : modes(embedding(H)))
eigenvalues(E::DelayEmbedding) = E.Eigenvalues
eigenvalues(H::HAVOKmodel) = eigenvalues(embedding(H))
eigenseries(E::DelayEmbedding) = E.Eigenseries
eigenseries(H::HAVOKmodel) = eigenseries(embedding(H))
derivative(H::HAVOKmodel) = H.NumericalDerivative
coefficients(H::HAVOKmodel) = H.RegressionCoefficient
fulltspan(H::HAVOKmodel) = H.sim.fulltspan
forcing(H::HAVOKmodel; plotQ=false) = return (plotQ ? forcing_plot(H) : eigenseries(H)[:,rank(H)])
function dynamics(H::HAVOKmodel; vars=1:rank(H), tspan=fulltspan(H), plotQ=false)
    if plotQ
        # Plots real data, svd approx, and HAVOK model prediction
        return dynamics_plot(H; vars=vars, tspan=tspan)
    else
        @unpack dt, Embedding, sim, r = H
        @unpack Eigenseries, Eigenmodes, Eigenvalues = Embedding
        @unpack fulltspan, sol = sim

        base = (Eigenmodes)*(Diagonal(Eigenvalues))

        range = map(x->Int(round((x - fulltspan[1]) / dt) + 1), tspan)      # idx range of solution
        HAVOKData = base[:,1:r-1]*sol'

        return mean_off_diagonal(HAVOKData, range)
    end
end


function HAVOKanalysis(timeseries::AbstractArray, dt::AbstractFloat;
    d=0.0, q=0, r=0,
    der_method="CentralDifferenceFourthOrder",
    reg_method="SequentialLeastSquares",
    interceptQ=false, selectQ=MinBIC(), λ=0)

    println("\n----------------------------------")
    println("||    HAVOK Analysis Summary    ||")
    println("----------------------------------")

    #=== EMBEDDING ===#

    q, r, Embedding = embed(timeseries; q=q, r=r, d=d)

    #=== LINEAR MODEL ===#

    # Numerical derivative dX estimator and linear regression dX = X * β fit
    β, dX, range = fit(Embedding,dt;
        der_method=der_method,      # Could be a function or default method
        reg_method=reg_method,      # Could be a function or default method
        interceptQ=interceptQ,      # Bias fit boolean flag
        λ=λ,selectQ=selectQ)        # Params of default regression methods

    #=== INTERMITTENTLT FORCED MODEL ===#

    # From ControlSystems ⨰ = A x + B u; y = C x + D u
    sys = intermittently_forced_state_space_system(β)

    #=== SOLVE MODEL ===#

    # HAVOKsim with whole time trajectory
    sol = sim(sys, range, Embedding, dt)

    # Create and return HAVOKmodel object
    return HAVOKmodel(timeseries, d, q, r, dt, Embedding, dX, β, sys, sol)
end


function fit(Embedding::DelayEmbedding, dt::AbstractFloat;
    der_method="CentralDifferenceFourthOrder",  # default method from DOI: 10.1038/s41467-017-00030-8
    reg_method="SequentialLeastSquares",        # default method from DOI: 10.1038/s41467-017-00030-8
    interceptQ=false,λ=0,selectQ=MinBIC())      # params of solvers and bias fit boolean flag

    @unpack Eigenseries = Embedding
    r = size(Eigenseries,2)

    #=== SPARSE LINEAR REGRESSION ===#

    # Build design matrix and its numerical derivative approximation
    range, dX = NumericalDifferentiation(Eigenseries, dt; der_method=der_method)
    X = Eigenseries[range,:]

    norms = [norm(X[:,i]) for i in 1:r]     # columns norms
    X = X./norms'   # broadcasts norms across respective columns => columns normalized

    println("Using $der_method the numerical derivative ∈ $(size(dX)) was computed.")

    # β fit in dX = X * β
    β = RegressionSolve(X,dX;reg_method=reg_method,interceptQ=interceptQ,λ=λ,selectQ=selectQ)

    println("Using $reg_method the regression coefficients ∈ $(size(β)) were computed.")

    println("Quality of fit measured with pearson correlations for each derivative:")
    for i in 1:r
        println("$i: $(cor(dX[:,i],X*β[1+(end-r):end,i]))")
    end

    # Brunton normalizes the coefficients?!, maybe because the
    # elimination of the entries in V where derivative didnt exist
    # de-normalizes V but why apply the normalization on β?
    length(β)==r*r ? (β = β./norms) : (β = β[1+(end-r):end,:]./norms) # Broadcasts across rows

    return β, dX, range
end


function intermittently_forced_state_space_system(β::Array{<:AbstractFloat,2})
    # Remember that y=X*β therefore (y[i,:])'=β'*(X[i,:])'
    s, r = size(β)
    βview = view(β, 1+(s-r):s, 1:s)' # transposed and intercept_col dropped

    # Define the matrices state A, input B , output C, and feedforward D.
    A = βview[1:r-1,1:r-1]
    B = βview[1:r-1,r]          # forcing
    C = Diagonal(ones(r-1))
    D = zeros(size(B))

    println("A model of the form ⨰ = A x + B u; y = C x + D u is proposed.\n")

    # Define sys from ControlSystems ⨰ = A x + B u; y = C x + D u
    return ss(A,B,C,D)
end


function sim(sys::StateSpace{R,Array{R,2}}, range::UnitRange{Int}, Embedding::DelayEmbedding{R}, dt::R) where R <: AbstractFloat
    @unpack Eigenseries = Embedding
    tspan = dt * range          # time span of simulation
    x0 = Eigenseries[range[1], 1:end-1] # initial Condition
    u = Eigenseries[range, end]         # external Forcing
    sol, t, full_sol = lsim(sys, u, tspan, x0=x0)   # solve

    # Create and return HAVOKsim object
    return HAVOKsim(sol, tspan, x0, u)
end
