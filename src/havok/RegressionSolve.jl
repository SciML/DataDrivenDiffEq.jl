function RegressionSolve(X::Array{<:AbstractFloat,2}, y::Array{<:AbstractFloat,2}; reg_method="LASSO", interceptQ::Bool=false, λ::Real=0, selectQ=MinBIC(),n::Int=1)
    r = size(y)[2]

    # Allocates memory of bias term in regression when flag is true
    if interceptQ
        β = zeros(r+1,r)
    else
        β = zeros(r,r)
    end

    # If it is string checks if is a registered method, else der_method must be manually crafted method
    if typeof(reg_method) == String
        if reg_method=="LASSO"  # FOR SOME REASON LASSO DOESNT WORK WELL FOR HAVOK
            for i in 1:r        # Derivative of one dimension at a time
                β[:,i] = coef(fit(LassoPath, X, y[:,i], intercept=interceptQ), select=selectQ) # interceptQ fits unpenalized bias
            end
        end

        # Adds bias term manually as predictor... it wasnt added before since Pkg Lasso doesnt need it.
        if interceptQ
            X = hcat(normalize(ones(size(X)[1])),X)
        end

        if reg_method=="SequentialLeastSquares"
            # Derivative of one dimension at a time
            #=
            # Frist we define a set of variables and parameters
            @variables u[1:r]

            # Then we simply create a basis
            b = Basis(u, u)
            β = SInDy(X, y, b;  maxiter=1)
            =#
            for i in 1:r
                β[:,i] = SequentialLeastSquares(X, y[:,i], λ*i, n)
            end
        elseif reg_method=="LeastSquares"
            # Defaults to pseudoinverse
            β = X\y
        else
            println("Regression method $method unknown.")
        end
    else
        # Manually crafted method
        β = reg_method(X,y)
    end
    return β
end

# Compute sparse regression: sequential least squares
function SequentialLeastSquares(X::Array{<:AbstractFloat,2}, y::Array{<:AbstractFloat,1}, λ::Real, maxiter::Int)
    # initial guess: Least-squares
    β = X\y

    for k=1:maxiter
        # find small/big coefficients with λ as our sparsification knob.
        smallinds = findall(x->abs(x)<λ, β)
        biginds = findall(x->abs(x)>=λ, β)

        # threshold smallinds
        β[smallinds] = zeros(length(smallinds))

        # n is state dimension
        for i = 1:size(y, 2)
            # Regress dynamics onto remaining terms to find sparse Xi
            β[biginds,i] = X[:,biginds]\y[:,i];
        end
    end
    return β
end
