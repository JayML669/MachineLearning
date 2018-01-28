from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('ggplot')


def create_dataset(size, variance, step=1, correlation=None):
    """Create a dataset of arrays of x and y values"""
    val = 1
    ys = []
    for i in range(size):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step

    xs = [i for i in range(size)]

    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def best_fit_slope_yint(xs, ys):
    """Find the slope and y-intercept of the line of best fit (linear regression) of the dataset"""
    meanx = mean(xs)
    meany = mean(ys)

    meanxy = mean(xs*ys)

    meanx2 = mean([x**2 for x in xs])

    numerator = (meanx*meany) - (meanxy)
    denominator = (meanx**2) - (meanx2)

    m = numerator/denominator

    b = meany - (m*meanx)

    return m, b


def evaluate_model(model, original_line):
    """Evaluate the linear regression model using the coefficient of determination"""

    meany = mean(original_line)

    squared_error_model = sum((model-original_line)**2)
    squared_error_mean = sum((meany-ys)**2)
    r_squared = 1 - (squared_error_model / squared_error_mean)

    return r_squared


# Creating the dataset of x and y values
xs, ys = create_dataset(50, 25, 3, 'pos')

# Get the line of best fit(linear regression model) of the dataset
m, b = best_fit_slope_yint(xs, ys)

# Create an array of values using the linear regression model and pass it in to evaluate the model
model = [(m*x+b) for x in xs]
coefficient_of_determination = evaluate_model(model, ys)

print(m, b, coefficient_of_determination)

# creates x and y values that are used to plot line of best fit
xplot = np.linspace(0, len(xs), 100)
y = m*xplot+b

# Plot the data and line of best fit
plt.plot(xplot, y)
plt.scatter(xs, ys)
plt.show()
