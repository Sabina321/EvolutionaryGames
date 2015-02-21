import numpy as np

def strategy_fitness_at_x(w,i,x):
    row = w[i]
    return np.array([row[j]*x[j] for j in range(row.size)])
    
def average_fitness_at_x(w,x):
    w_bar = 0
    for i in range(len(x)):
        w_bar = w_bars+np.dot(x,strategy_fitness_at_x(w,i,x))
    return w_bar
