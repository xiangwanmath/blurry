#Problem 1.2


import numpy as np

v = np.arange(1, 2346, 2)

dimension = v.shape[0]
print(f"The dimension of vector v is: {dimension}")

l1_norm = np.linalg.norm(v, ord=1)
print(f"The L1 norm of v is: {l1_norm}")

l2_norm = np.linalg.norm(v, ord=2)
print(f"The L2 norm of v is: {l2_norm}")

inf_norm = np.linalg.norm(v, ord=np.inf)
print(f"The infinity norm of v is: {inf_norm}")


#Problem 1.5


import numpy as np
from matplotlib import pyplot as plt
import math

def newton(f, df, x0, errorTolerance=1e-8, maxIterations=100, demoMode=False):
    x = x0
    iterations = 0
    error = float('inf')
    
    while error > errorTolerance and iterations < maxIterations:
        fx = f(x)
        dfx = df(x)
        
        if dfx == 0:
            raise ValueError("Derivative is zero. Newton's method fails.")
        
        x_new = x - fx / dfx
        error = abs(x_new - x)
        x = x_new
        iterations += 1
        
        if demoMode:
            print(f"Iteration {iterations}: x = {x}, f(x) = {f(x)}, error = {error}")
    
    return x, error, iterations

def f_1(x): 
    return x**3+x-1 

def Df_1(x): 
    return 3*x**2+1  

(root, errorEstimate, iterations) = newton(f_1, Df_1, x0=0., errorTolerance=1e-8, demoMode=True)
print()
print(f"The root is approximately {root}")
print(f"The estimated absolute error is {errorEstimate:0.3}")
print(f"The backward error is {abs(f_1(root)):0.3}")
print(f"This required {iterations} iterations")


#1.6


import numpy as np

def f_1(x):
    return 4*x**4 - 6*x**2 + 3*x - 11/4

def Df_1(x):
    return 16*x**3 - 12*x + 3

def newton(f, df, initial_x, errorTolerance=1e-8, maxIterations=100, demoMode=False):
    solution = []
    x = initial_x
    iterations = 0
    error = float('inf')

    while error > errorTolerance and iterations < maxIterations:
        fx = f(x)
        dfx = df(x)

        if dfx == 0:
            raise ValueError("Derivative is zero. Newton's method fails.")

        x_new = x - fx / dfx
        error = abs(x_new - x)
        x = x_new
        iterations += 1
        solution.append(x_new)

        if demoMode:
            print(f"Iteration ({iterations}): x = {x:.8f}, f(x) = {fx:.8f}, error = {error:.8f}")

    if demoMode:
        print(f"The solutions are: {solution}")

    return x, solution

if __name__ == "__main__":
    results = []
    initial_guesses = np.linspace(-10, 10, 5)

    for x0 in initial_guesses:
        try:
            root, history = newton(f_1, Df_1, x0, demoMode=True)
            print(f"\nFound root starting from x0 = {x0:.2f}: {root:.8f} after {len(history)} iterations.")
            results.append(root)
        except ValueError as e:
            print(f"\nError starting from x0 = {x0:.2f}: {e}")

    print("\nAll found roots:", np.unique(np.round(results, 6)))


# ## Problem 1.7a


import numpy as np
from matplotlib import pyplot as plt
import math

def newton(f, df, x0, errorTolerance=1e-8, maxIterations=100, demoMode=False):
    x = x0
    iterations = 0
    error = float('inf')
    
    while error > errorTolerance and iterations < maxIterations:
        fx = f(x)
        dfx = df(x)
        
        if dfx == 0:
            raise ValueError("Derivative is zero. Newton's method fails.")
        
        x_new = x - fx / dfx
        error = abs(x_new - x)
        x = x_new
        iterations += 1
        
        if demoMode:
            print(f"Iteration {iterations}: x = {x}, f(x) = {f(x)}, error = {error}")
    
    return x, error, iterations

def f_1(x): 
    return math.exp(x) + 2*x 

def Df_1(x): 
    return math.exp(x) + 2   

(root, errorEstimate, iterations) = newton(f_1, Df_1, x0=0., errorTolerance=1e-8, demoMode=True)
print()
print(f"The mininmum x value is approximately {root}")
print(f"The estimated absolute error is {errorEstimate:0.3}")
print(f"The backward error is {abs(f_1(root)):0.3}")
print(f"This required {iterations} iterations")


# ## Problem 1.7b

import numpy as np
from matplotlib import pyplot as plt
import math

def newton(f, df, x0, errorTolerance=1e-8, maxIterations=100, demoMode=False):
    x = x0
    iterations = 0
    error = float('inf')

    while error > errorTolerance and iterations < maxIterations:
        fx = f(x)
        dfx = df(x)
        
        if dfx == 0:
            raise ValueError("Derivative is zero")
        
        x_new = x - fx / dfx
        error = abs(x_new - x)
        x = x_new
        iterations += 1
        
        if demoMode:
            print(f"Iteration {iterations}: x = {x}, f(x) = {f(x)}, error = {error}")
    
    return x, error, iterations

def f_2(x): 
    return math.cos(x)-.2*x

def Df_2(x): 
    return -math.sin(x)-.2  

(root, errorEstimate, iterations) = newton(f_2, Df_2, x0=-1., errorTolerance=1e-8, demoMode=True)
print()
print(f"The first root is approximately {root}")
if Df_2(root)<0: 
    print(f"{root} is a max")
if Df_2(root)>0:
    print(f"{root} is a min")

(root2, errorEstimate2, iterations2) = newton(f_2, Df_2, x0=-5, errorTolerance=1e-8, demoMode=True)
print()
print(f"The second root is {root2}")
if Df_2(root2)<0: 
    print(f"{root2} is a max")
if Df_2(root2)>0:
    print(f"{root2} is a min")

(root3, errorEstimate3, iterations3) = newton(f_2,Df_2, x0=5, errorTolerance=1e-8, demoMode=True)
print()
print(f"The third root is approximately {root3}")
if Df_2(root3)<0: 
    print(f"{root3} is a max")
if Df_2(root3)>0:
    print(f"{root3} is a min")


# ## Problem 1.8a


import numpy as np

def f(x):
    x=x[0]
    return math.exp(x) + x**2

def gradient (x):
    x=x[0]
    d_x = math.exp(x) + 2*x
    return np.array([d_x])

def gradient_descent(initial_x = np.array([0.0]), learning_rate = .0001, n_iterations = 1000, tolerance=1e-6):
    x = np.array(initial_x, dtype=float)
    history = [x.copy()]
    for i in range(n_iterations):
        grad = gradient(x)
        x = x - learning_rate * grad
        history.append(x.copy())

        if np.linalg.norm(grad) < tolerance:
            print(f"Converged after {i+1} iterations.")
            break

    return x, history
initial_x = np.array([0.0])
learning_rate = .1
n_iterations = 10000
x_min_gd, history = gradient_descent(initial_x, learning_rate, n_iterations)
min_value_gd = f(x_min_gd)

print(f"Minimum found at x (GD): [{x_min_gd[0]:.4f}]")
print(f"Minimum value (GD): {min_value_gd:.4f}")


# ## Problem 1.8b


import numpy as np

def f(x):
    x=x[0]
    return math.sin(x)-.1*x**2

def gradient (x):
    x=x[0]
    d_x = math.cos(x)-.2*x
    return np.array([d_x])

def gradient_descent(initial_x = np.array([0.0]), learning_rate = .0001, n_iterations = 1000, tolerance=1e-6):
    x = np.array(initial_x, dtype=float)
    history = [x.copy()]
    for i in range(n_iterations):
        grad = gradient(x)
        x = x - learning_rate * grad
        history.append(x.copy())

        if np.linalg.norm(grad) < tolerance:
            print(f"Converged after {i+1} iterations.")
            break

    return x, history
initial_x = np.array([0.0])
learning_rate = 1
n_iterations = 10000
x_min_gd, history = gradient_descent(initial_x, learning_rate, n_iterations)
min_value_gd = f(x_min_gd)

print(f"Minimum found at x (GD): [{x_min_gd[0]:.4f}]")
print(f"Minimum value (GD): {min_value_gd:.4f}")


# ## Problem 1.9


import numpy as np

def f(theta):
    theta1 = theta[0]
    theta2 = theta[1]
    return theta1**2 + 4*theta2**2 + 2*theta1*theta2 + 3*theta1 - theta2 + 5

def gradient(theta):
    theta1 = theta[0]
    theta2 = theta[1]
    d_theta1 = 2*theta1 + 2*theta2 + 3
    d_theta2 = 8*theta2 + 2*theta1 - 1
    return np.array([d_theta1, d_theta2])

def gradient_descent(initial_theta, learning_rate, n_iterations, tolerance=1e-6):
    theta = np.array(initial_theta, dtype=float)
    history = [theta.copy()]

    for i in range(n_iterations):
        grad = gradient(theta)
        theta = theta - learning_rate * grad
        history.append(theta.copy())

        if np.linalg.norm(grad) < tolerance:
            print(f"Converged after {i+1} iterations.")
            break

    return theta, history

initial_theta = np.array([0.0, 0.0])
learning_rate = 0.02
n_iterations = 1000

theta_min_gd, history = gradient_descent(initial_theta, learning_rate, n_iterations)
min_value_gd = f(theta_min_gd)

print(f"Minimum found at theta (GD): [{theta_min_gd[0]:.4f}, {theta_min_gd[1]:.4f}]")
print(f"Minimum value (GD): {min_value_gd:.4f}")

theta_min_analytic = np.array([-13/6, 2/3])
min_value_analytic = f(theta_min_analytic)

print(f"Minimum found at theta (Analytic): [{theta_min_analytic[0]:.4f}, {theta_min_analytic[1]:.4f}]")
print(f"Minimum value (Analytic): {min_value_analytic:.4f}")


# ## Problem 1.10

import numpy as np

def f(theta):
    theta1=theta[0]
    theta2=theta[1]
    theta3=theta[2]
    theta4=theta[3]
    return theta1**2 + 3*theta2**2 + 2*theta3**2 + theta4**2 + theta1*theta2 - 2*theta2*theta3 + theta3*theta4 - 4*theta1 + theta2 + 2*theta3 - theta4 + 10

def gradient (theta):
    theta1=theta[0]
    theta2=theta[1]
    theta3=theta[2]
    theta4=theta[3]
    d_theta1 = 2*theta1+theta2-4
    d_theta2 = 6*theta2+theta1-2*theta3+1
    d_theta3 = 4*theta3-2*theta2+theta4+2
    d_theta4 = 2*theta4+theta3-1
    return np.array([d_theta1, d_theta2, d_theta3, d_theta4])

def gradient_descent(initial_theta = np.array([0.0,0.0,0.0,0.0]), learning_rate = .0001, n_iterations = 1000, tolerance=1e-10):
    theta = np.array(initial_theta, dtype=float)
    history = [theta.copy()]
    for i in range(n_iterations):
        grad = gradient(theta)
        theta = theta - learning_rate * grad
        history.append(theta.copy())

        if np.linalg.norm(grad) < tolerance:
            print(f"Converged after {i+1} iterations.")
            break

    return theta, history
initial_theta = np.array([0.0, 0.0, 0.0, 0.0])
learning_rate = .2
n_iterations = 100000
theta_min_gd, history = gradient_descent(initial_theta, learning_rate, n_iterations)
min_value_gd = f(theta_min_gd)

print(f"Minimum found at theta (GD): [{theta_min_gd[0]:.4f}, {theta_min_gd[1]:.4f},  {theta_min_gd[2]:.4f}, {theta_min_gd[3]:.4f}]")
print(f"Minimum value (GD): {min_value_gd:.4f}")
    
    


# ## Problem 1.11


import numpy as np

def midpoint_quadrature (f, a, b, n):
    h=(b-a)/n
    integral_sum=0
    for i in range(n):
        midpoint = a+(i+.5)*h
        integral_sum+=f(midpoint)
    print(f"The midpoint integration is {h*integral_sum}")
    return h*integral_sum

def f(x):
    return x**2

result = midpoint_quadrature(f, a=0, b=2, n=12)
print(f"The final midpoint integration result is: {result}")


