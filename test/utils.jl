@info "Starting utilities tests"
@testset "Utilities" begin
    t = collect(-2:0.01:2)
    U = [cos.(t).*exp.(-t.^2) sin.(2*t)]
    S = Diagonal([2.; 3.])
    V = [sin.(t).*exp.(-t) cos.(t)]
    A = U*S*V'
    σ = 0.5
    Â = A + σ*randn(401, 401)
    n_1 = norm(A-Â)
    B = optimal_shrinkage(Â)
    optimal_shrinkage!(Â)
    @test norm(A-Â) < n_1
    @test norm(A-B) == norm(A-Â)

    X = randn(3, 100)
    Y = randn(3, 100)
    k = 3

    @test AIC(k, X, Y) == 2*k-2*log(sum(abs2, X- Y))
    @test AICC(k, X, Y) == AIC(k, X, Y)+ 2*(k+1)*(k+2)/(size(X)[2]-k-2)
    @test BIC(k, X, Y) == -2*log(sum(abs2, X -Y)) + k*log(size(X)[2])
    @test AICC(k, X, Y, likelihood = (X,Y)->sum(abs, X-Y)) == AIC(k, X, Y, likelihood = (X,Y)->sum(abs, X-Y))+ 2*(k+1)*(k+2)/(size(X)[2]-k-2)

    # Numerical derivatives
    function lorenz(u,p,t)
        x, y, z = u
        ẋ = 10.0*(y - x)
        ẏ = x*(28.0-z) - y
        ż = x*y - (8/3)*z
        return [ẋ, ẏ, ż]
    end
    u0 = [1.0;0.0;0.0]
    tspan = (0.0,50.0)
    dt = 0.005
    prob = ODEProblem(lorenz,u0,tspan)
    sol = solve(prob, Tsit5(), saveat = dt)

    X = Array(sol)
    DX = similar(X)
    for (i, xi) in enumerate(eachcol(X))
        DX[:,i] = lorenz(xi, [], 0.0)
    end

    windowSize, polyOrder = 9, 4
    halfWindow = Int(ceil((windowSize+1)/2))
    DX_sg = similar(X)
    DX_sg_cropped = similar(X[:,halfWindow+1:end-halfWindow])
    X_cropped = similar(X[:,halfWindow+1:end-halfWindow])
    for i =1:size(X,1)
        DX_sg[i,:] = savitzky_golay(X[i,:], windowSize, polyOrder, deriv=1, dt=dt, crop=false)
        X_cropped[i,:], DX_sg_cropped[i,:] = savitzky_golay(X[i,:], windowSize, polyOrder, deriv=1, dt=dt)
    end
    DX_sg2 = savitzky_golay(X, windowSize, polyOrder, deriv=1, dt=dt, crop=false)
    X_cropped2, DX_sg_cropped2 =  savitzky_golay(X, windowSize, polyOrder, deriv=1, dt=dt)
    @test(DX_sg2 == DX_sg)
    DX_sg = DX_sg[:,halfWindow+1:end-halfWindow]
    @test X_cropped == X[:,halfWindow+1:end-halfWindow]
    @test DX_sg_cropped == DX_sg
    @test X_cropped2 == X[:,halfWindow+1:end-halfWindow]
    @test DX_sg_cropped2 == DX_sg

    DX = DX[:,halfWindow+1:end-halfWindow]
    @test isapprox(DX_sg, DX, rtol=1e-2)

    # Sampling
    X = randn(Float64, 2, 100)
    t = collect(0:0.1:9.99)
    Y = randn(size(X))
    xt = burst_sampling(X, 5, 10)
    @test 10 <= size(xt)[end] <= 60
    @test all([any(xi .≈ X) for xi in eachcol(xt)])
    xt, tt = burst_sampling(X, t, 5, 10)
    @test all(diff(tt) .> 0.0)
    @test size(xt)[end] == size(tt)[end]
    @test all([any(xi .≈ X) for xi in eachcol(xt)])
    @test !all([any(xi .≈ Y) for xi in eachcol(xt)])
    xs, ts = burst_sampling(X, t, 2.0, 1)
    @test all([any(xi .≈ X) for xi in eachcol(xs)])
    @test size(xs)[end] == size(ts)[end]
    @test ts[end]-ts[1] ≈ 2.0
    X2n = subsample(X, 2)
    t2n = subsample(t, 2)
    @test size(X2n)[end] == size(t2n)[end]
    @test size(X2n)[end] == Int(round(size(X)[end]/2))
    @test X2n[:, 1] == X[:, 1]
    @test X2n[:, end] == X[:, end-1]
    @test all([any(xi .≈ X) for xi in eachcol(X2n)])
    xs, ts = subsample(X, t, 0.5)
    @test size(xs)[end] == size(ts)[end]
    @test size(xs)[1] == size(X)[1]
    @test all(diff(ts) .≈ 0.5)
    # Loop this a few times to make sure it's right
    @test_nowarn for i in 1:20
        xs, ts = burst_sampling(X, t, 2.0, 1)
        xs, ts = subsample(X, t, 0.5)
    end
end
