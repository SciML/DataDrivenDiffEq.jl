let f(x) = [exp10(first(x)-last(x)); x[1]*x[3]+x[1]*x[2]; sum(x); prod(x[1:2])/x[3]]
    f(x::AbstractMatrix) = reduce(hcat, map(f, eachcol(x)))
    x = randn(3, 1000)
    y = f(x)
    s = DataDrivenSurrogate(f, x)

    for i in 1:size(y, 1)
        @test norm(s[i](x) - y[i:i, :]) < 1e-10
        @test DataDrivenDiffEq.depth(s[i]) == (i == 3 ? 1 : 3)
        @test DataDrivenDiffEq.get_operator(s[i]) == (i == 3 ? identity : *)
        @test DataDrivenDiffEq.is_linear(s[i]) == (i == 3 ? true : false)
    end
end
