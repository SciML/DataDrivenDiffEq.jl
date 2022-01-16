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

@recipe function probplot(p::AbstractDataDrivenProblem)
    plotins, ylab = gather_plot_information(p)

    layout := (length(plotins), 1)
    isdirec = is_direct(p)

    for (i, pins) in enumerate(plotins)
        yi, xi, lab = pins
        lab = isdirec ? lab : lab*"(t)"
        @series begin
            label --> reduce(hcat, map(j->lab*subscriptnumber(j), 1:size(xi,1)))
            
            ylabel --> lab
            
            if length(plotins) == i
                xlabel --> ylab
            end

            subplot := i
            yi, permutedims(xi)
        end
    end

end