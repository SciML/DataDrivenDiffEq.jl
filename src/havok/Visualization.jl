import Plots.plot3d, Plots.plot, Plots.heatmap

# TODO
# ADD COLORING PARAMETRIZED WITH TIME TO PHASE PLANE
# ADD PAIRED COLORINGS BETWEEN DATA AND MODEL PRED IN EIGENTIMESERIES
# ADD HAVOKmodel METHODS TO EXTRACT INFO
# ADD CROSSVALIDATION ROUTINES
# ADD HISTOGRAM LAYOUT AND DISCOVER WHY EIGENSERIES EMPIRIRCAL DISTRIBUTIONS SEEMS GEOMETRICALLY SIMILAR

# Plots phase plane
function PPplane(H::HAVOKmodel; vars=(1,2,3), title="HAVOK Attractor", label="HAVOKmodel", tspan=H.sim.tspan)
    if 0 in vars && length(vars) > 3
        eigentimeseries(H; vars=vars, title=title, tspan=tspan)    # Plots time vs selected Eigenseries
    elseif !(length(vars) in [2,3])
        error("Selection of vars not supported")
    end

    range = map(x->Int(round((x - H.sim.tspan[1]) / H.dt) + 1), tspan)  # idxs of simulation requested
    sol = [H.sim.sol[range,i] for i in vars if i!=0]                    # simulation requested

    if length(vars)==3
        if !(0 in vars)
            # Plots 3d phase plane
            plot3d([H.sim.sol[range,i] for i in vars if i!=0]..., title=title, label=label,
                xlabel="V$(vars[1])", ylabel="V$(vars[2])", zlabel="V$(vars[3])")
        else
            # Plots 2d phase combined with time
            plot3d(tspan, sol..., title=title, label=label,
                xlabel="time", ylabel="V$(vars[2])", zlabel="V$(vars[3])")
        end

    else
        # Plots 2d phase plane
        plot(sol..., title=title, label="HAVOK simulation",
            xlabel="V$(vars[1])", ylabel="V$(vars[2])")
    end
end

# Plots one or more eigenseries
function eigentimeseries(H::HAVOKmodel; vars=(0,1), title="", tspan=H.sim.tspan)
    if (0 in vars) # time in plot vars
        range = map(x->Int(round((x - H.sim.tspan[1]) / H.dt) + 1), tspan)  # idxs of simulation requested
        sol = [H.sim.sol[range,i] for i in vars if i!=0]                    # simulation requested

        # Plots time vs selected Eigenseries
        plot(tspan, sol, title=title, label=reshape(["V$i" for i in vars if i!=0], (1,length(vars)-1)), xlabel="t", ylabel="V")
    else            # time not in plot vars therefore it must be a phase plane
        PPplane(H; vars=vars, title=title, tspan=tspan)          # delegates work to PPplane above
    end
end

# Plots forcing timeseries
forcing(H::HAVOKmodel) = plot(H.Embedding.Eigenseries[:,H.r], title="Forcing Timeseries", label="V$(H.r)")

# Controler to quickly plot and compare data attractor to predictions
function plot(H::HAVOKmodel; vars=(1,2,3), tspan=H.sim.tspan, layout=(1,1))

    range = map(x->Int(round((x - H.sim.tspan[1]) / H.dt) + 1), tspan)  # idx range of solution
    X = [H.Embedding.Eigenseries[range,i] for i in vars if i!=0]        # Data embedded attractor


    if !(0 in vars) # time not in plot vars
        # Plots HAVOK simulation
        p1 = PPplane(H; vars=vars, tspan=tspan, title="")

        # Overlapping display
        if layout==(1,1)
            return plot!(X..., label="Data Attractor", title="Dimeomorphic Attractor Comparison", line=:dash)

        # Other displays
        else
            # Plots data attractor
            if length(vars) == 3
                p2 = plot(X..., label="Data Attractor",
                        xlabel="V$(vars[1])", ylabel="V$(vars[2])", zlabel="V$(vars[3])")
            else
                p2 = plot(X..., label="Data Attractor",
                        xlabel="V$(vars[1])", ylabel="V$(vars[2])")
            end
            return plot(p1,p2,layout=layout, plot_title="Dimeomorphic Attractor Comparison")
        end

    else # time in plot vars
        p1 = eigentimeseries(H; vars=vars, tspan=tspan, title="Eigenseries Comparison")

        # Overlapping display
        if layout==(1,1)
            return plot!(tspan, X, line=:dash, label=reshape(["Data V$i" for i in vars if i!=0],(1,length(vars)-1)))

        # Other displays
        else
            # Plots data attractor
            p2 = plot(tspan, X, label=reshape(["Data V$i" for i in vars if i!=0],(1,length(vars)-1)))
            return plot(p1,p2,layout=layout, plot_title="Eigenseries Comparison")
        end
    end
end
plot3d(H::HAVOKmodel; vars=(1,2,3), tspan=H.sim.tspan, layout=(1,1)) = plot(H; vars=vars, tspan=tspan, layout=layout)

# Plots heatmap of regression coefficients from dX = X * β
function heatmap(H::HAVOKmodel; title="Regression Solution")
    heatmap(H.RegressionCoefficient, yflip=true, title=title)
end

# Plots Eigenmodes from Embedding
function modes(H::HAVOKmodel; vars=1:H.r)
    @unpack Eigenmodes = H.Embedding
    return plot(1:size(Eigenmodes,1), Eigenmodes[:,vars],
            title="Normalized U Modes", label=reshape(["U$i" for i in vars], (1, length(vars))))
end

# Plots Eigenseries empirial distribution with corresponding gaussian fit
function eigenseries_dist(H::HAVOKmodel, vars)
    @unpack Eigenseries = H.Embedding

    # Plot a histogram of the Eigenseries
    histogram(Eigenseries[:,vars], normalize=:pdf, color=:darkorange, label=reshape(["Histogram V$var" for var in vars],(1,length(vars))))

    # Plot a gaussian with sample mean and variance
    for var in vars
        σ = std(Eigenseries[:,var])
        plt_range = -3*σ:σ/50:3*σ
        gaussianD(μ, σ) = x -> 1/(exp((((x-μ)/σ)^2)/2)*(σ*sqrt(2*pi)))
        gauss_vec = map(gaussianD(0, σ), plt_range)
        plot!(-3*σ:σ/50:3*σ, gauss_vec, color=:black, line=(:dash,4), title="Eigenseries Statistics", label="Gaussian Fit $var")
    end
    plot!()
end

# Empirical distribution of forcing timeseries
forcing_dist(H::HAVOKmodel) = eigenseries_dist(H, H.r)

# Allow to compare data, svd and predicted dynamics
function dynamics(H::HAVOKmodel; vars=1:H.r, tspan=H.sim.tspan)
    @unpack Eigenseries, Eigenmodes, Eigenvalues = H.Embedding

    range = map(x->Int(round((x - H.sim.tspan[1]) / H.dt) + 1), tspan)      # idx range of solution

    # Plots real data, svd approx, and HAVOK model prediction
    base = (Eigenmodes)*(Diagonal(Eigenvalues))

    Data = (base*(Eigenseries)')
    Data = vcat(Data[:,1], Data[end,2:end])
    plot(tspan, Data[range], title="Data Approximation", label="Real Data", line=(4,:solid))

    SVData = base[:,vars]*(Eigenseries[:,vars])'
    SVData = mean_off_diagonal(SVData, range)
    plot!(tspan, SVData, label="SVD $(H.r)-approx", line=(3,:dash), color=:black)

    HAVOKData = base[:,1:H.r-1]*H.sim.sol'
    HAVOKData = mean_off_diagonal(HAVOKData, range)
    plot!(tspan, HAVOKData, label="HAVOK pred", line=(4,:dot), color=:orange)
end
