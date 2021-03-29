# The basis definition
@variables u[1:2]
basis = Basis([polynomial_basis(u, 5); sin.(u); cos.(u)], u)

function pendulum(u, p, t)
    x = u[2]
    y = -9.81sin(u[1]) - 0.1u[2]^3 - 0.2 * cos(u[1])
    return [x;y]
end

u0 = [0.99π; -1.0]
tspan = (0.0, 20.0)
dt = 0.1
prob = ODEProblem(pendulum, u0, tspan)
sol = solve(prob, Tsit5(), saveat=dt)

X = sol[:,:]
t = sol.t
DX = similar(sol[:,:])
for (i, xi) in enumerate(eachcol(sol[:,:]))
    DX[:,i] = pendulum(xi, [], 0.0)
end

@testset "Ideal data" begin


    dd_prob = ContinuousDataDrivenProblem(
        X, t, DX = DX
        )


    opts = [
    STLSQ(1e-2), ADMM(1e-2, 0.1), SR3(1e-2, SoftThreshold()),
    SR3(1e-4)
    ]


    for opt in opts
        res = solve(dd_prob, basis, opt, maxiter = 10000)
        m = metrics(res)
        @test m.Sparsity == 4
        @test m.Error ./ size(X, 2) < 3e-1
        @test m.AICC < 0.0
    end
end


X = X .+ 1e-1*randn(size(X))

@testset "Noisy data" begin

    dd_prob_noisy = ContinuousDataDrivenProblem(
        X, t, SigmoidKernel()
        )

    opts = [
        STLSQ(1e-1:5e-1:1e3), ADMM(1e-1:5e-1:1e2, 0.1), SR3(1e-1:5e-1:1e2, SoftThreshold()),
        SR3((1e-1:5e-1:1e2).^2, 50.0, HardThreshold()), SR3(1e-1:5e-1:1e3, ClippedAbsoluteDeviation())
    ]


    for opt in opts
        res = solve(dd_prob_noisy, basis, opt, maxiter = 50000, denoise = true, normalize = true)
        @test m.Sparsity <= 4
        @test m.Error ./ size(X, 2) < 5e-1
        @test m.AICC < 0.0
    end

end
