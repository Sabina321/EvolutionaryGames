import numpy as np
from sympy import *

def strategy_fitness_at_x(w,i,x):
    row = w[i]
    return np.dot(row,x)
    
#w is the payoff matrix 
#x is the state of shares
def average_fitness_at_x(w,x):
    w_bar = 0
    for i in range(len(x)):
        w_bar = w_bar+np.dot(x[i],strategy_fitness_at_x(w,i,x))
    return w_bar




def get_delta_line(w,i,j,k):
    si,sj,sk=symbols('si,sj,sk')
    a = [si,sj,sk]
    if k==0:
        a = [1-sj-sk,sj,sk]
    elif k==1:
        a = [si,1-si-sk,sk]
    elif k==2:
        a = [si,sj,1-si-sj]
    wi = np.dot(a,w[i])
    wj = np.dot(a,w[j])
    z = wi-wj
    return [diff(z,si),diff(z,sj),diff(z,sk),[x if type(x)==Integer else 0 for x in z.args][0]]

def get_interior_equilibrium_3_by_3(w):
    delta_w_12 = get_delta_line(w,0,1,2)
    delta_w_23 = get_delta_line(w,1,2,2)
    A = np.array([delta_w_12[:2],delta_w_23[:2]])
    b = np.array([delta_w_12[3],delta_w_23[3]])
    shares = np.linalg.solve(A,-1*b)
    final_shares = [shares[0],shares[1],(1-(shares[0]+shares[1]))]
    if 1-sum(final_shares)<1e-16 and all(0<x<1 for x in final_shares):
        return final_shares
    else:
        return "No internal equilibrium"

