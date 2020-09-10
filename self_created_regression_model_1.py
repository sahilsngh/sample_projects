from statistics import mean
import random
import numpy as np
import matplotlib.pyplot as plt
# remember the function is y = m*x + b for linear regression model.
# this is sample data used to create model.
# xs = np.array([1, 3, 5, 6, 9, 10], dtype=np.float)
# ys = np.array([5, 6, 9, 8, 7, 11], dtype=np.float)

def dataset_generation(hm, variance, step=2, correlation=False):
    val = 1
    ys = []
    for y in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        if correlation and correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float), np.array(ys, dtype=np.float)

def slope_of_line_and_intercept(xs, ys):
    m = (((mean(xs) * mean(ys)) - mean(xs * ys)) /
         (mean(xs)**2 - mean(xs**2)))
    b = mean(ys) - (m * mean(xs))
    return m, b

def squared_error(ys_og, ys_line):
    return sum((ys_og-ys_line)**2)

def coefficient_of_determination(ys_og, ys_line):
    ys_line_mean = [mean(ys_og) for y in ys_og]
    squared_error_reg = squared_error(ys_og, ys_line)
    squared_error_line = squared_error(ys_og, ys_line_mean)
    return 1 - (squared_error_reg / squared_error_line)

# tweak the values inside to check and see how the model performes.
xs, ys = dataset_generation(1000, 50, 2, correlation='pos')

m, b = slope_of_line_and_intercept(xs, ys)
reg_line = [(m*x+b) for x in xs]
r_squared = coefficient_of_determination(ys, reg_line)
x_input = 12
y_predict = m * x_input + b
print(r_squared)

plt.scatter(x_input, y_predict, s=100, color='g')
plt.scatter(xs, ys, color='black')
plt.plot(xs, reg_line, color='r')
plt.show()



