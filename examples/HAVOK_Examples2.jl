using DataDrivenDiffEq
using DelimitedFiles, Plots, Statistics, Interpolations

# LORENZ
# Brunton parameters
x_measurements = readdlm(".\\data\\x_lorenz.csv", ',')[:]
q = 100                         # Delays
dt = 0.001                      # Equidistant timestep of measurements

# Lorenz HAVOK Model
lorenz = HAVOKanalysis(x_measurements,dt;q=q,der_method="CentralDifferenceFourthOrder",reg_method="SequentialLeastSquares",interceptQ=false,λ=0)

# Plots
plot3d(lorenz, vars=(2,3,1), layout=(1,1))
plot3d(lorenz, vars=(2,3,1), layout=(1,2))
plot(lorenz, vars=(0,1,2,3), tspan=185:dt:199, layout=(1,1))
heatmap(lorenz)
forcing(lorenz)
plot(PearsonCorrelates(lorenz), title="Pearson Correlations", ylabel="v(r) correlation with HAVOK solution", xlabel="r", label="correlation")
modes(lorenz; vars=1:lorenz.r)
forcing_dist(lorenz)
eigenseries_dist(lorenz, 1:9, layout=(3,3))
dynamics(lorenz, tspan=185:dt:199)

# RÖSSLER
# Brunton parameters
x_measurements = readdlm(".\\data\\x_rossler.csv", ',')[:]
q = 100                         # Delays
dt = 0.001                      # Equidistant timestep of measurements

# rössler HAVOK Model
rössler = HAVOKanalysis(x_measurements,dt;q=q,der_method="CentralDifferenceFourthOrder",reg_method="SequentialLeastSquares",interceptQ=true,λ=0)

# Plots
plot3d(rössler, vars=(1,3,2), layout=(1,1))
plot3d(rössler, vars=(1,3,2), layout=(1,2))
plot(rössler, vars=(0,1,2,3), tspan=400:dt:490, layout=(1,1))
heatmap(rössler)
forcing(rössler)
plot(PearsonCorrelates(rössler), title="Pearson Correlations", ylabel="v(r) correlation with HAVOK solution", xlabel="r", label="correlation")
modes(rössler; vars=1:rössler.r)
forcing_dist(rössler)
eigenseries_dist(rössler, 1:9, layout=(3,3))
dynamics(rössler, tspan=400:dt:490)

# measles data
data = readdlm(".\\data\\nycmeas.csv", ',')[:]
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
plot3d(measles, vars=(1,2), layout=(1,1))
plot3d(measles, vars=(1,2), layout=(1,2))
plot(measles, vars=(0,1,2,3), tspan=100:dt:300, layout=(1,1))
heatmap(measles)
forcing(measles)
plot(PearsonCorrelates(measles), title="Pearson Correlations", ylabel="v(r) correlation with HAVOK solution", xlabel="r", label="correlation")
modes(measles; vars=1:measles.r)
forcing_dist(measles)
eigenseries_dist(measles, 1:9, layout=(3,3))
dynamics(measles, tspan=100:dt:300)


# ELECTROCARDIOGRAM
# Consists of three columns: time, measurement1, measurement2 respectively. measurement2 seems cleaner
measurements = readdlm(".\\data\\ECG.txt", '\t')
q = 26        # Delays (Brunton choses 25 and retains 67.4 of the energy and selects measurement1)
dt = mean([abs(measurements[i,1] - measurements[i+1,1]) for i=1:size(measurements,1)-1])    # Avg dt ≈ 0.004
x_measurements = measurements[:,3] .- mean(measurements[:,3])

# ECG HAVOK Model
ECG = HAVOKanalysis(x_measurements,dt;q=q,der_method="CentralDifferenceFourthOrder",reg_method="SequentialLeastSquares",interceptQ=true,λ=0)

# Plots
plot3d(ECG, vars=(1,3,2), layout=(1,1))
plot3d(ECG, vars=(1,3,2), layout=(1,2))
plot(ECG, vars=(0,1,2,3), tspan=175:dt:179, layout=(1,1))
heatmap(ECG)
forcing(ECG)
plot(PearsonCorrelates(ECG), title="Pearson Correlations", ylabel="v(r) correlation with HAVOK solution", xlabel="r", label="correlation")
modes(ECG; vars=1:ECG.r)
forcing_dist(ECG)
eigenseries_dist(rössler, 1:4, layout=(2,2))
dynamics(ECG, tspan=175:dt:179)


#SLEEP EEG
# Brunton parameters
x_measurements = readdlm(".\\data\\sleepEEG100000.csv", ',')[:]
q = 1000                        # Delays
r = 4
dt = 0.01                       # Equidistant timestep of measurements

# Sleep EEG HAVOK Model
sleepEEG = HAVOKanalysis(x_measurements,dt;q=q,r=r,der_method="CentralDifferenceFourthOrder",reg_method="SequentialLeastSquares",interceptQ=true,λ=0)

# Plots
plot3d(sleepEEG, vars=(1,2), layout=(1,1))
plot3d(sleepEEG, vars=(1,2), layout=(1,2))
plot(sleepEEG, vars=(0,1,2,3), tspan=900:dt:950, layout=(1,1))
heatmap(sleepEEG)
forcing(sleepEEG)
plot(PearsonCorrelates(sleepEEG), title="Pearson Correlations", ylabel="v(r) correlation with HAVOK solution", xlabel="r", label="correlation")
modes(sleepEEG; vars=1:sleepEEG.r)
forcing_dist(sleepEEG)
eigenseries_dist(rössler, 1:4, layout=(2,2))
dynamics(sleepEEG, tspan=900:dt:950)
