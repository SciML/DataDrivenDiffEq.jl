# ADD FFT AND LEVANT DIFFERENTIATION AS DEFAULT METHOD

# Numerical Derivative method controller
function NumericalDifferentiation(V::Array{<:AbstractFloat,2}, dt::AbstractFloat; der_method="CentralDifferenceFourthOrder")
    m = size(V,1)

    # If it is string checks if is a registered method, else der_method must be function
    if typeof(der_method) == String

        # Consists in storing the numerical derivative of V in y
        # and identifying the range of elements in V for which
        # the derivative exists
        if der_method=="CentralDifferenceSecondOrder"
            y = CentralDifferenceSecondOrder(V, dt)
            range = 2:m-2
        elseif der_method=="CentralDifferenceFourthOrder"
            y = CentralDifferenceFourthOrder(V, dt)
            range = 3:m-3
        elseif der_method=="CentralDifferenceSixthOrder"
            y = CentralDifferenceSixthOrder(V, dt)
            range = 4:m-4
        elseif der_method=="BSplineQuadratic"
            tspan = dt:dt:size(V)[1]*dt
            y = zeros(size(V))
            range = 1:m
            for i in 1:size(V)[2]
                itp = interpolate(V[:,i], BSpline(Quadratic(Free(OnGrid()))))
                itp = Interpolations.scale(itp, dt:dt:(size(V)[1]*dt))
                y[:,i] = [Interpolations.gradient(itp,i)[1] for i in tspan]
            end
        elseif der_method=="BSplineCubic"
            tspan = dt:dt:size(V)[1]*dt
            y = zeros(size(V))
            range = 1:m
            for i in 1:size(V)[2]
                itp = CubicSplineInterpolation(tspan, V[:,i])
                y[:,i] = [Interpolations.gradient(itp,i)[1] for i in tspan]
            end
        else
            println("Numerical derivative method $method unknown.")
        end
    else
        # Manually crafted method
        range, y = der_method(V, dt)
    end

    return range, y
end

# Compute Derivatives (2th order central difference)
function CentralDifferenceSecondOrder(V::Array{<:AbstractFloat,2}, dt::AbstractFloat)
    m = size(V)[1]
    dV = zeros(m-3,size(V)[2])
    for i=2:m-2
        # coefficients can be found in https://en.wikipedia.org/wiki/Finite_difference_coefficient
        dV[i-1,:] = (1/(2*dt))*(V[i+1,:]-V[i-1,:])
    end
    return dV
end

# Compute Derivatives (4th order central difference)
function CentralDifferenceFourthOrder(V::Array{<:AbstractFloat,2}, dt::AbstractFloat)
    m = size(V)[1]
    dV = zeros(m-5,size(V)[2])
    for i=3:m-3
        # coefficients can be found in https://en.wikipedia.org/wiki/Finite_difference_coefficient
        dV[i-2,:] = (1/(12*dt))*(-V[i+2,:]+8*V[i+1,:]-8*V[i-1,:]+V[i-2,:])
    end
    return dV
end

# Compute Derivatives (6th order central difference)
function CentralDifferenceSixthOrder(V::Array{<:AbstractFloat,2}, dt::AbstractFloat)
    m = size(V)[1]
    dV = zeros(m-7,size(V)[2])
    for i=4:m-4
        # coefficients can be found in https://en.wikipedia.org/wiki/Finite_difference_coefficient
        dV[i-3,:] = (1/(60*dt))*(V[i+3,:]-9*V[i+2,:]+45*V[i+1,:]-45*V[i-1,:]+9V[i-2,:]-V[i-3,:])
    end
    return dV
end
