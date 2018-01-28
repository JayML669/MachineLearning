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


m, b = best_fit_slope_yint(xs, ys)
print(m, b)

x = np.linspace(0, len(xs), 100)
y = m*x+b

plt.plot(x, y)
plt.scatter(xs, ys)
plt.show()
