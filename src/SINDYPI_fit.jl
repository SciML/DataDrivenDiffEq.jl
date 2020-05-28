##FIT FUNCTION

function fit!(X::AbstractArray, Y::AbstractArray, maxiter, lambda, Polyorder, Trigorder)
    using LinearAlgebra
    # Compute awesome stuff here
    s = []
    for i = 1:1:size(X,2)
        push!(s, string("x", string(i)))
    end

number_states_guess = size(X, 2)

Store_Result = Array{Any}(undef, number_states_guess)
Index_nonzero = Array{Any}(undef, number_states_guess)
Xi = Array{Any}(undef, number_states_guess)


 for w = 1:number_states_guess
    npoly = Polyorder[w]
    ntrig = Trigorder[w]

    nvar = size(X, 2)

    basis = ones(size(X,1), 1)

    if npoly >=1
      basis = hcat(basis, X)
    end

    if npoly >= 2
        for i in 1:nvar
            for j in i:nvar
                basis = hcat(basis, (X[:, i] .* X[:, j]) )
            end
        end
    end


    if npoly >= 3
        for i in 1:nvar
            for j in i:nvar
                for k in j:nvar
                basis = hcat(basis, (X[:,i] .* X[:, j] .* X[:, k]) )
            end
            end
        end
    end


    if npoly >= 4
        for i in 1:nvar
            for j in i:nvar
                for k in j:nvar
                    for l in k:nvar
                basis =  hcat(basis, (X[:,i].* X[:, j] .* X[:,k] .* X[:,l]))
            end
            end
        end
    end
end

if npoly >= 5
    for i in 1:nvar
        for j in i:nvar
            for k in j:nvar
                for l in k:nvar
                    for m in l:nvar
            basis =  hcat(basis, (X[:,i].* X[:, j] .* X[:,k] .* X[:,l] .* X[:,m]))
        end
        end
    end
end
end
end

if npoly >= 6
    for i in 1:nvar
        for j in i:nvar
            for k in j:nvar
                for l in k:nvar
                    for m in l:nvar
                        for n in m:nvar
            basis =  hcat(basis, (X[:,i].* X[:, j] .* X[:,k] .* X[:,l] .* X[:,m] .* X[:,n]))
        end
        end
    end
end
end
end
end

if ntrig >=1
    basis = hcat(basis, sin.(X))
    basis = hcat(basis, cos.(X))
end

if ntrig >=2
    basis = hcat(basis, cos.(X) .* cos.(X))
end

if ntrig >=3
    for i in 1:nvar
        for j in 1:nvar
            basis = hcat(basis, (X[:, i] .* X[:, i] .* sin.(X[:, j]) .* cos.(X[:, j]) ))
        end
    end
end

if ntrig >=4
    for i in 1:nvar
        for j in 1:nvar
            basis = hcat(basis, (X[:, i] .* X[:, i] .* sin.(X[:, j])))
        end
    end

    for i in 1:nvar
        for j in 1:nvar
            basis = hcat(basis, (X[:, i] .* X[:, i] .* cos.(X[:, j])))
        end
    end

    for i in 1:nvar
        for j in 1:nvar
            basis = hcat(basis, (sin.(X[:, i]) .* cos.(X[:, j]) ))
        end
    end
end

    pin = size(basis,2)

    basis_noderivative = deepcopy(basis[:, 1:pin])

    basis_deep = zeros(size(basis,1), 2*size(basis,2))

    basis_deep[:, 1:size(basis,2)] = basis_noderivative

    for k=1:pin
        basis_deep[:,pin + k] = Y[:, w].* basis_noderivative[:, k]
    end

    Right_train = zeros(size(basis_deep, 1), size(basis_deep, 2)-1)

    for i = 1:size(basis_noderivative,2)
        Right_train[:, i] = basis_deep[:,i]
    end

    for i = size(basis_noderivative,2)+1:size(basis_deep, 2)-1
        Right_train[:, i] = basis_deep[:,i+1]
    end

    Left_train = Y[:, w]

    normLib = zeros(1, size(Right_train,2))

    Theta = deepcopy(Right_train)

        for norm_k=1:size(Theta,2)
                normLib[norm_k] = norm(Theta[:,norm_k]);
                Theta[:,norm_k] = Theta[:,norm_k]/normLib[norm_k];
        end

     Epsilon = zeros(size(Theta,2),1)
     ldiv!(Epsilon, qr(Theta), Left_train)

    for i in 1:maxiter
        index_remove = abs.(Epsilon) .<= lambda
        Epsilon[index_remove] .= zero(eltype(Epsilon))

        for j in 1:1
            index_retain = @. !index_remove[:,j]
            Epsilon[index_retain, j] = Theta[:, index_retain] \ Left_train[:,j]
        end
    end

        for norm_k=1:length(Epsilon)
            Epsilon[norm_k,:] = Epsilon[norm_k,:]/normLib[norm_k];
        end

  Xi[w] = Epsilon

  nvar = length(s)
  table = []
  #push!(table, "")

  push!(table, "1")

 if npoly >=1
  for i in 1:nvar
      push!(table, s[i])
  end
end

  if npoly >= 2
      for i in 1:nvar
          for j in i:nvar
              push!(table, string(s[i], s[j]))
          end
      end
  end


  if npoly >= 3
      for i in 1:nvar
          for j in i:nvar
              for k in j:nvar
                push!(table, string(s[i], s[j], s[k]))
          end
          end
      end
  end


  if npoly >= 4
      for i in 1:nvar
          for j in i:nvar
              for k in j:nvar
                  for l in k:nvar
                    push!(table, string(s[i], s[j], s[k], s[l]))

          end
          end
      end
  end
end

if npoly >= 5
  for i in 1:nvar
      for j in i:nvar
          for k in j:nvar
              for l in k:nvar
                  for m in l:nvar
                push!(table, string(s[i], s[j], s[k], s[l], s[m]))

      end
      end
  end
end
end
end

if npoly >= 6
  for i in 1:nvar
      for j in i:nvar
          for k in j:nvar
              for l in k:nvar
                  for m in l:nvar
                      for n in m:nvar
                push!(table, string(s[i], s[j], s[k], s[l], s[m], s[n]))

      end
      end
  end
end
end
end
end

if ntrig >=1
 for i in 1:nvar
     push!(table, string("sin", "", s[i]))
 end

 for i in 1:nvar
     push!(table, string("cos", "", s[i]))
 end
end

if ntrig >=2
 for i in 1:nvar
     push!(table, string("cos^{2}", "", s[i]))
 end
end

if ntrig >=3
  for i in 1:nvar
      for j in 1:nvar
          push!(table, string(s[i], "*", s[i], "sin", "", s[j], "cos", "", s[j]))
      end
  end
end

if ntrig >=4
  for i in 1:nvar
      for j in 1:nvar
          push!(table, string(s[i], "*", s[i], "sin", "", s[j]))
      end
  end

  for i in 1:nvar
      for j in 1:nvar
          push!(table, string(s[i], "*", s[i], "cos", "", s[j]))
      end
  end

  for i in 1:nvar
      for j in 1:nvar
          push!(table, string("sin", "", s[i], "cos", "", s[j]))
      end
  end
end


tablecopy = deepcopy(table)

for i in 1:length(table)-1
  push!(table, string("dot", "(", s[w], ")" , "", tablecopy[i+1]))
end


String_table = hcat(table, Xi[w])

Store_Result[w] = String_table

Index_nonzero[w] = findall(x->x !=0, Store_Result[w][:,2])

ODE_Index = Index_nonzero[w]

ODE_Strings = Store_Result[w][:,1][ODE_Index]

ODE_values = Xi[w][ODE_Index]

ODE_Print = []
push!(ODE_Print, "dot", s[w])
push!(ODE_Print, "", "=")


global count = 0
for i = 1:length(ODE_Strings)
    global count
 if count >0
    if string(round(ODE_values[i]))[1] == '-'
        pop!(ODE_Print)
    end
  end

    push!(ODE_Print, string(string(round(ODE_values[i], digits = 6)), ODE_Strings[i]))
    push!(ODE_Print, "", "+")

    count += 1
end

pop!(ODE_Print)

ODE_Print = join(ODE_Print)

println("The differential equation for ", s[w], " is"  )

println(ODE_Print)
end
    #return iterations
end

#TEST 1: Michaelis - Menten
jx = 0.6
Vm = 1.5
Km = 0.3

func(u,p,t) = jx - (Vm*u/(Km + u))

u0 = 1/2
tspan = (0.0,3.0)
dt = 0.01
prob = ODEProblem(func,u0,tspan)

sol = DifferentialEquations.solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8, saveat = dt)

X = sol.u
DX = jx .- (Vm*X./(Km .+ X))

Polyorder =  Int64[1];
Trigorder = Int64[0];

fit!(X, DX, 10, 0.1, Polyorder, Trigorder)

#TEST EXAMPLE 2: Stiff Oregonator
using DifferentialEquations, Plots
function oregonator!(du,u,p,t)
    s,w,q = p
    #du = zeros(3, 1)
    du[1] = s*(u[2]-u[1]*u[2] + u[1] - q*u[1]*u[1])
    du[2] = (1/s)*(- u[2] - u[1]*u[2] + u[3])
    du[3] = w*(u[1] - u[3])
    return [du[1], du[2], du[3]]
end

u0 = [1.0,2.0,3.0]

p = (77.27,0.161,8.375e-6) # we could also make this an array, or any other sequence type!

tspan = (0.0,360.0)
dt = 0.005

prob = ODEProblem(oregonator!, u0, tspan, p)

sol = solve(prob, Rodas5(), saveat = dt)

X = Array(sol)

DX = similar(X)

du = zeros(3,1)

for (i, xi) in enumerate(eachcol(X))
    DX[:,i] = oregonator!(du, xi, p, 0.0)
end

X = X'

DX = DX'


Polyorder =  Int64[2 2 1];
Trigorder = Int64[0 0 0];


fit!(X, DX, 10, 0.1, Polyorder, Trigorder)

##TEST EXAMPLE 3: Cart Pendulum
# Setup
initial = [0.3, 0, 1.0, 0]
tspan = (0,16.0)
dt = 0.001;

 m = 1
 M = 1
 L = 1
 g = 9.81


# Define the function
function Cart_Pendulum!(du, u,p,t)
    du[1] = u[3]
    du[2] = u[4]
    du[3] = -((M +m)*g*sin(u[1]) +m*(L^2)*sin(u[1])*cos(u[1])*u[3]*u[3] )/(L*L*(M + m -m*cos(u[1])*cos(u[1])))
    du[4] = (m*L*L*sin(u[1])*u[3]*u[3] + m*g*sin(u[1])*cos(u[1]))/(L*(M + m -m*cos(u[1])*cos(u[1])))
    return [du[1] du[2] du[3] du[4]]
end

#Pass to solvers
prob = ODEProblem(Cart_Pendulum!, initial, tspan)

sol = solve(prob, Tsit5(), saveat = dt)


X = Array(sol)
DX = similar(X)
du = zeros(4,1)
for (i, xi) in enumerate(eachcol(X))
    DX[:,i] = Cart_Pendulum!(du, xi, [], 0.0)
end
X = X'
DX = DX'

Polyorder =  Int64[1 1 0 0];
Trigorder = Int64[0 0 3 4];

fit!(X, DX, 10, 0.1, Polyorder, Trigorder)
