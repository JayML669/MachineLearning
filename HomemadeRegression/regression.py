from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)


def best_fit_slope_yint(xs, ys):
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
    meany = mean(original_line)

    squared_error_model = sum((model-original_line)**2)
    squared_error_mean = sum((meany-ys)**2)
    r_squared = 1 - (squared_error_model / squared_error_mean)

    return r_squared


m, b = best_fit_slope_yint(xs, ys)

model = [(m*x+b) for x in xs]
coefficient_of_determination = evaluate_model(model, ys)

print(m, b, coefficient_of_determination)

xplot = np.linspace(0, len(xs), 100)
y = m*xplot+b

plt.plot(xplot, y)
plt.scatter(xs, ys)
plt.show()
