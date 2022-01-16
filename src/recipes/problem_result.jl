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
    
    t = x.t 
    
    ylab = is_direct(x) ? "Sample ID" : "t"

    returns = []

    for (xi, si) in zip([X, DX, Y, U], ["x", "\U02202\U0209C"*"x", "y", "u"])
        isempty(xi) && continue
        push!(returns, (t, xi, si))
    end

    return returns, ylab
end

function gather_plot_information(x::AbstractDataDrivenSolution)
    p = get_problem(x)
    t = p.t 
    
    t = is_discrete(p) ? t[2:end] : t
    ylab = is_direct(p) ? "Sample ID" : "t"
    
    Y = get_target(p)
    outsym = is_direct(p) ? "y" : (is_discrete(p) ? "x" : "\U02202\U0209C"*"x")
    est_sym = outsym * "\U00302"
    X, _, ts, u = get_oop_args(p)
    Ŷ = x.basis(X, parameters(x), ts, u)
    return [(t, Y, outsym), (t, Ŷ, est_sym)], ylab
end

## Problem

@recipe function probplot(p::AbstractDataDrivenProblem)
    plotins, ylab = gather_plot_information(p)

    layout := (length(plotins), 1)
    isdirec = is_direct(p)

    for (i, pins) in enumerate(plotins)
        yi, xi, lab = pins
        suff = isdirec ? "" : "(t)"
        @series begin
            label --> reduce(hcat, map(j->lab*subscriptnumber(j)*suff, 1:size(xi,1)))
            
            ylabel --> lab
            
            if length(plotins) == i
                xlabel --> ylab
            end

            subplot := i
            yi, permutedims(xi)
        end
    end

end

# Step 2...

## Solution !

@recipe function resplot(x::AbstractDataDrivenSolution)
    plotins, xlab = gather_plot_information(x)
    isdirec = is_direct(get_problem(x))
    layout := (2,1)
    t, Y, lab = plotins[1]

    
    suff = isdirec ? "" : "(t)"

    @series begin
        label --> nothing #reduce(hcat, map(j->lab*subscriptnumber(j), 1:size(Y, 1)))
        ylabel --> lab
        subplot := 1
        seriestype := :scatter
      #  xlabel --> xlab
        t, permutedims(Y)
    end

    _ , Ŷ, lab_ = plotins[2]

    @series begin
        label --> reduce(hcat, map(j->lab_*subscriptnumber(j)*suff, 1:size(Y, 1)))
        ylabel --> lab_
        subplot := 1
        seriestype := :path
       # xlabel --> xlab
        t, permutedims(Ŷ)
    end

    elab = lab*suff*"-"*lab_*suff

    @series begin
        label --> reduce(hcat, map(j->"e"*subscriptnumber(j)*suff, 1:size(Y, 1)))
        ylabel --> elab
        subplot := 2
        xlabel --> xlab
        t, permutedims(Y-Ŷ)
    end

end