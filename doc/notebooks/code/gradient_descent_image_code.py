import numpy as np
import matplotlib.pyplot as plt

# Defining a simple quadratic function f(x) = x^2
def f(x):
    return x ** 2

# Derivative of the function f(x) = x^2
def df(x):
    return 2 * x

# Initial parameter and learning rate
x_start = 4.5
learning_rate = 0.15
# Gradient descent iterations
iterations = 15

# Storing the progression of x values for plotting
x_progression = [x_start]
y_progression = [f(x_start)]

# Performing the gradient descent
for _ in range(iterations):
    x_gradient = df(x_progression[-1])
    x_next = x_progression[-1] - learning_rate * x_gradient
    x_progression.append(x_next)
    y_progression.append(f(x_next))

# Creating a range of x values for plotting the function
x_values = np.linspace(-5, 5, 100)
y_values = f(x_values)

# Plotting the function f(x)
plt.plot(x_values, y_values, label=r'$f(x) = x^2$')

# Plotting the gradient descent progression
plt.scatter(x_progression, y_progression, color='red', zorder=5)
plt.plot(x_progression, y_progression, linestyle='--', color='red', label='Gradient Descent')

# Adding details to the plot
plt.title('2D Graph Illustrating Gradient Descent')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()
