using DataDrivenDiffEq
using DelimitedFiles, Plots, Statistics, Interpolations

# LORENZ
# Brunton parameters
x_measurements = readdlm("C:\\Users\\dan_y\\Documents\\TesisSS\\NonlinearDynamics\\HAVOK\\data\\x_lorenz.csv", ',')[:]
q = 100                         # Delays
dt = 0.001                      # Equidistant timestep of measurements

# Lorenz HAVOK Model
lorenz = HAVOKanalysis(x_measurements,dt;q=q,der_method="CentralDifferenceFourthOrder",reg_method="SequentialLeastSquares",interceptQ=true,λ=0)

# Plots
attractor1v2_img = plot3d(lorenz, vars=(2,3,1), layout=(1,1))
attractor1v1_img = plot3d(lorenz, vars=(2,3,1), layout=(1,2))
eigenseries_img = plot(lorenz, vars=(0,1,2,3), tspan=185:dt:199, layout=(1,1))
heatmap_img = heatmap(lorenz)
forcing_img = forcing(lorenz)
pearsoncorrelates_img = plot(PearsonCorrelates(lorenz), title="Pearson Correlations", ylabel="v(r) correlation with HAVOK solution", xlabel="r", label="correlation")
modes_img = modes(lorenz; vars=1:lorenz.r)
forcingstats_img = forcing_dist(lorenz)
reconstruction_img = dynamics(lorenz, tspan=185:dt:199)


# RÖSSLER
# Brunton parameters
x_measurements = readdlm("C:\\Users\\dan_y\\Documents\\TesisSS\\NonlinearDynamics\\HAVOK\\data\\x_rossler.csv", ',')[:]
q = 100                         # Delays
dt = 0.001                      # Equidistant timestep of measurements

# rössler HAVOK Model
rössler = HAVOKanalysis(x_measurements,dt;q=q,der_method="CentralDifferenceFourthOrder",reg_method="SequentialLeastSquares",interceptQ=true,λ=0)

# Plots
attractor1v2_img = plot3d(rössler, vars=(1,3,2), layout=(1,1))
attractor1v1_img = plot3d(rössler, vars=(1,3,2), layout=(1,2))
eigenseries_img = plot(rössler, vars=(0,1,2,3), tspan=400:dt:490, layout=(1,1))
heatmap_img = heatmap(rössler)
forcing_img = forcing(rössler)
pearsoncorrelates_img = plot(PearsonCorrelates(rössler), title="Pearson Correlations", ylabel="v(r) correlation with HAVOK solution", xlabel="r", label="correlation")
modes_img = modes(rössler; vars=1:rössler.r)
forcingstats_img = forcing_dist(rössler)
reconstruction_img = dynamics(rössler, tspan=400:dt:490)

# measles data
data = readdlm("C:\\Users\\dan_y\\Documents\\TesisSS\\NonlinearDynamics\\HAVOK\\data\\nycmeas.csv", ',')[:]
itp = interpolate(data, BSpline(Quadratic(Reflect(OnCell()))))
dt = 0.1                        # Equidistant timestep of measurements
aug_data = [itp(t+1) for t in 0:dt:(length(data)-1)]
plot(aug_data, label="NYC Weekly measles Infections")


# MEASLES
# Brunton parameters
q = 50                          # Delays

# Lorenz HAVOK Model
measles = HAVOKanalysis(aug_data,dt;q=q,der_method="CentralDifferenceFourthOrder",reg_method="SequentialLeastSquares",interceptQ=false,λ=0.0)

# Plots
attractor1v2_img = plot3d(measles, vars=(1,2), layout=(1,1))
attractor1v1_img = plot3d(measles, vars=(1,2), layout=(1,2))
eigenseries_img = plot(measles, vars=(0,1,2,3), tspan=100:dt:300, layout=(1,1))
heatmap_img = heatmap(measles)
forcing_img = forcing(measles)
pearsoncorrelates_img = plot(PearsonCorrelates(measles), title="Pearson Correlations", ylabel="v(r) correlation with HAVOK solution", xlabel="r", label="correlation")
modes_img = modes(measles; vars=1:measles.r)
forcingstats_img = forcing_dist(measles)
reconstruction_img = dynamics(measles, tspan=100:dt:300)


# ELECTROCARDIOGRAM
# Consists of three columns: time, measurement1, measurement2 respectively. measurement2 seems cleaner
measurements = readdlm("C:\\Users\\dan_y\\Documents\\TesisSS\\NonlinearDynamics\\HAVOK\\data\\ECG.txt", '\t')
q = 26        # Delays (Brunton choses 25 and retains 67.4 of the energy and selects measurement1)
dt = mean([abs(measurements[i,1] - measurements[i+1,1]) for i=1:size(measurements,1)-1])    # Avg dt ≈ 0.004
x_measurements = measurements[:,3] .- mean(measurements[:,3])

# ECG HAVOK Model
ECG = HAVOKanalysis(x_measurements,dt;q=q,der_method="CentralDifferenceFourthOrder",reg_method="SequentialLeastSquares",interceptQ=true,λ=0)

# Plots
attractor1v2_img = plot3d(ECG, vars=(1,3,2), layout=(1,1))
attractor1v1_img = plot3d(ECG, vars=(1,3,2), layout=(1,2))
eigenseries_img = plot(ECG, vars=(0,1,2,3), tspan=175:dt:179, layout=(1,1))
heatmap_img = heatmap(ECG)
forcing_img = forcing(ECG)
pearsoncorrelates_img = plot(PearsonCorrelates(ECG), title="Pearson Correlations", ylabel="v(r) correlation with HAVOK solution", xlabel="r", label="correlation")
modes_img = modes(ECG; vars=1:ECG.r)
forcingstats_img = forcing_dist(ECG)
reconstruction_img = dynamics(ECG, tspan=175:dt:179)


#SLEEP EEG
# Brunton parameters
x_measurements = readdlm("C:\\Users\\dan_y\\Documents\\TesisSS\\NonlinearDynamics\\HAVOK\\data\\sleepEEG100000.csv", ',')[:]
q = 1000                        # Delays
r = 4
dt = 0.01                       # Equidistant timestep of measurements

# Sleep EEG HAVOK Model
sleepEEG = HAVOKanalysis(x_measurements,dt;q=q,r=r,der_method="CentralDifferenceFourthOrder",reg_method="SequentialLeastSquares",interceptQ=true,λ=0)

# Plots
attractor1v2_img = plot3d(sleepEEG, vars=(1,2), layout=(1,1))
attractor1v1_img = plot3d(sleepEEG, vars=(1,2), layout=(1,2))
eigenseries_img = plot(sleepEEG, vars=(0,1,2,3), tspan=900:dt:950, layout=(1,1))
heatmap_img = heatmap(sleepEEG)
forcing_img = forcing(sleepEEG)
pearsoncorrelates_img = plot(PearsonCorrelates(sleepEEG), title="Pearson Correlations", ylabel="v(r) correlation with HAVOK solution", xlabel="r", label="correlation")
modes_img = modes(sleepEEG; vars=1:sleepEEG.r)
forcingstats_img = forcing_dist(sleepEEG)
reconstruction_img = dynamics(sleepEEG, tspan=900:dt:950)
