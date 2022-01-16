# Based on StackOverflow
function subscriptnumber(i::Int)
    if i < 0
        c = [Char(0x208B)]
    else
        c = []
    end
    for d in reverse(digits(abs(i)))
        push!(c, Char(0x2080+d))
    end
    return join(c)
end

function gather_plot_information(x::AbstractDataDrivenProblem{N, C, D}) where {N,C, D}
    
    X = getfield(x, :X)
    Y = getfield(x, :Y)
    DX = getfield(x, :DX)
    U = getfield(x, :U)
    
    t = has_timepoints(x) ? x.t : collect(one(N):size(X,2))
    
    ylab = has_timepoints(x) ? "t" : "ID"

    returns = []

    for (xi, si) in zip([X, DX, Y, U], ["x", "\U02202\U0209C"*"x", "y", "u"])
        isempty(xi) && continue
        push!(returns, (t, xi, si))
    end

    return returns, ylab
end

@recipe function probplot(p::AbstractDataDrivenProblem)
    plotins, ylab = gather_plot_information(p)

    layout := (length(plotins), 1)

    for (i, pins) in enumerate(plotins)
        yi, xi, lab = pins
        @series begin
            label --> reduce(hcat, map(j->lab*subscriptnumber(j), 1:size(xi,1)))
            ylabel --> lab
            if length(plotins) == i
                xlabel --> ylab
            end
            #seriestype := :path
            subplot := i
            yi, permutedims(xi)
        end
    end

end

## DiscreteDataDrivenProblem

#
#@recipe function probplot(p::AbstractDiscreteProb{N, false}) where N
#    X = p.X
#    Y = p.Y
#    U = p.U
#
#    if has_timepoints(p)
#        t = p.t
#
#    end
#
#    # Collect the data
#    layout := (3, 1)
#
#    @series begin
#        label --> reduce(hcat, map(i->"x"*subscriptnumber(i), 1:size(X,1)))
#        ylabel --> "y(t)"
#        seriestype := :path
#        subplot := 1
#        t, permutedims(X)
#    end
#
#    @series begin
#        label --> reduce(hcat, map(i->"u"*subscriptnumber(i), 1:size(X,1)))
#        ylabel --> "u(t)"
#        seriestype := :path
#        subplot := 2
#        t, permutedims(U)
#        
#    end
#
#    @series begin
#        label --> reduce(hcat, map(i->"\U02202\U0209C"*"x"*subscriptnumber(i), 1:size(X,1)))
#        ylabel --> "\U02202\U0209C"*"x(t)"
#        seriestype := :path
#        subplot := 3
#        xlabel --> "t"
#    
#        t, permutedims(DX)
#    end
#
#    #t, permutedims(X)
#end
#
### ContinuousDataDrivenProblem
#
#@recipe function probplot(p::AbstractContProb{N, true}) where N
#    # Collect the data
#    t = p.t
#    X = p.X
#    DX = p.DX
#    layout := (2, 1)
#    @series begin
#        label --> reduce(hcat, map(i->"x"*subscriptnumber(i), 1:size(X,1)))
#        ylabel --> "x(t)"
#        seriestype := :path
#        subplot := 1
#        t, permutedims(X)
#    end
#
#    @series begin
#        label --> reduce(hcat, map(i->"\U02202\U0209C"*"x"*subscriptnumber(i), 1:size(X,1)))
#        ylabel --> "\U02202\U0209C"*"x(t)"
#        seriestype := :path
#        subplot := 2
#        xlabel --> "t"
#        t, permutedims(DX)
#    end
#
#    #t, permutedims(X)
#end
#
#
#@recipe function probplot(p::AbstractContProblem{N, false}) where N
#
#    # Collect the data
#    t = p.t
#    X = p.X
#    DX = p.DX
#    U = p.U
#    layout := (3, 1)
#
#    @series begin
#        label --> reduce(hcat, map(i->"x"*subscriptnumber(i), 1:size(X,1)))
#        ylabel --> "x(t)"
#        seriestype := :path
#        subplot := 1
#        t, permutedims(X)
#    end
#
#    @series begin
#        label --> reduce(hcat, map(i->"u"*subscriptnumber(i), 1:size(X,1)))
#        ylabel --> "u(t)"
#        seriestype := :path
#        subplot := 2
#        t, permutedims(U)
#        
#    end
#
#    @series begin
#        label --> reduce(hcat, map(i->"\U02202\U0209C"*"x"*subscriptnumber(i), 1:size(X,1)))
#        ylabel --> "\U02202\U0209C"*"x(t)"
#        seriestype := :path
#        subplot := 3
#        xlabel --> "t"
#    
#        t, permutedims(DX)
#    end
#
#    #t, permutedims(X)
#end
#
#t = 0:0.1:10.0
#X = randn(3, length(t))
#DX = randn(3, length(t))
#U = randn(2, length(t))
#prob = ContinuousDataDrivenProblem(X, t, DX = DX, U = U)

#plot(prob)