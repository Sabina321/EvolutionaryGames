import numpy as np
from sympy import *
import math
from random import random

def set_game(values):
    if(len(set([len(i) for i in values]))>1):
        return "matrix should be square"
    else:
        return values
def norms(A):
    return A

def w(share_stealers):
    return math.pow(10*share_stealers,2)

def share_when_average_fitnesses_are_the_same_two_strategies(f1,f2):
    return float(1.0*f2/(f1-f2))

def get_mean_over_time(time_range,share_stealers,A,k,update_A_at_t,p_detection,fine,p_pay_fine,bribe,price,penalty,service,use):
    matrix_of_interest = np.zeros((time_range,4))
    shares = [share_stealers,1-share_stealers]

    matrix_of_interest[0][0]=share_stealers
    cs=get_cost_stealing(A,k,p_detection,fine,p_pay_fine,bribe,share_stealers)
    cp=get_value_paying(price,penalty,share_stealers,service,use)
    w_bar = cs*(share_stealers)+cp*(1-share_stealers)
    
    norms_neighborhood = island(99,1,.1)
    p=.1
    s=.9
    #matrix_of_interest[0][1]=0
    w=[cs,cp]
    matrix_of_interest[0][1]=w_bar
    matrix_of_interest[0][2]=cp
    matrix_of_interest[0][3]=cs
    new_A=norms_neighborhood.mean()
    for t in range(1,time_range):
        
        s_new_stealer = replicator(shares,w,0)
        ##before we calculate new fitnesses 
        if(t%update_A_at_t==0):
            new_A =update_matrix(norms_neighborhood,p,s,1,10).mean()
         ##norms doesn't actually change get new A then call norms on A
         

        ##recalculate the fitnesses 
        
        cs=get_cost_stealing(new_A,k,p_detection,fine,p_pay_fine,bribe,s_new_stealer)
        cp=get_value_paying(price,penalty,s_new_stealer,service,use)
        w=[cs,cp]

        shares=[s_new_stealer,1-s_new_stealer]
        w_bar = w[0]*(s_new_stealer)+w[1]*(1-s_new_stealer)
        matrix_of_interest[t][0]=s_new_stealer
        matrix_of_interest[t][1]=w_bar
        matrix_of_interest[t][2]=cp
        matrix_of_interest[t][3]=cs
    return matrix_of_interest

def calculate_time_plus_one_values_simpler(grid,p,s):
    update_matrix = np.copy(grid)
    for i in range(len(grid)):
        for j in range(len(grid)):
            A = grid[i][j]
            alpha = calculate_alpha(grid,i,j)
            A_t_plus_one = A +((1-(alpha and A)) *((alpha*p)-((1-alpha)*s)))
            A_t_plus_one= round_extreme(A_t_plus_one)
            update_matrix[i][j]=A_t_plus_one
    return update_matrix

def round_extreme(value):
    if value<0:
        return 0
    elif value>1:
        return 1
    else:
        return value

def calculate_alpha(grid,i,j):
    #Von Neumann neighborhood
    alpha = 0
    update = 0
    #check for left neighbor
    if i-1>=0:

        alpha = alpha+grid[i-1][j]
        update = update+1
    #check for right neighbor
    if i+1<len(grid):
        alpha=alpha+grid[i+1][j]
        update = update+1
    #check for top neighbor
    if j-1>=0:
        alpha = alpha+grid[i][j-1]
        update = update+1
    #check for bottom neighbor
    if j+1<len(grid):
        alpha = alpha+grid[i][j+1]
        update = update+1
    if update==0:
        return grid[i][j]
    return alpha/update

def update_matrix(grid,p,s,k,time_range):
    #make movie later
    new_grid = np.copy(grid)
    for t in range(time_range):
        new_matrix =calculate_time_plus_one_values_simpler(new_grid,p,s)
        for i in range(len(grid)):
            for j in range(len(grid)):
                new_grid[i][j]=new_matrix[i][j]
    return new_grid

##little island of one
def island(n,steal,fanatic):
    grid = np.ones([n,n])
    for i in range(len(grid)):
        for j in range(len(grid)):
            g_temp=grid[i][j]*random()
            if not steal:
                if g_temp>.5:
                    g_temp=g_temp-.5
            else:
                if g_temp<.5:
                    g_temp=g_temp+.5
            grid[i][j]=g_temp
    ##set center to stealing
    
    grid[n/2][n/2]=fanatic
    return grid

def coin_flip():
    r = random()
    return (r<.5)-(r>=.5)

def neighborhood_centered_around_stealer(size,mean,variance):
    grid = np.zeros([size,size])
    for i in range(size):
        for j in range(size):
            grid[i][j]=round_extreme(mean + coin_flip()*random()*variance)

    return grid


def approaching_target(target_A,size,start_share,variance,p,s,k,time_range,limit,eps,play_island,stealer,fanatic):
    intermediates = []
    if play_island:
        neighborhood = island(size,stealer,fanatic)
    else:
        neighborhood = neighborhood_centered_around_stealer(size,start_share,variance)
    temp_grid = neighborhood
    intermediates.append(temp_grid)
    A = temp_grid.mean()
    t = 0
    As=[]
    while(round(abs(target_A-A),4)>=eps and t<limit):
        temp_grid = update_matrix(temp_grid,p,s,k,time_range)
        A = temp_grid.mean()
        As.append(A)
        intermediates.append(temp_grid)
        t=t+1
    return [A,t,As,intermediates,neighborhood]

#check 
def get_penalty_stealing(p_detection,fine,p_pay_fine,bribe):
    return p_detection*(fine*p_pay_fine+((1-p_pay_fine)*bribe))

def get_cost_paying(price,penalty,share_stealers):
    if(share_stealers==1):
        share_stealers=1-math.pow(10,-12)
    return (price*(1+ (float(1.0*penalty*share_stealers))/(1-share_stealers)))     

def get_value_paying(price,penalty,share_stealers,service,use):
    #value is benefits minus costs
    #the benefit is captured with the service term 
    return 1.0/(use*(get_cost_paying(price,penalty,share_stealers)))

def get_cost_stealing(A,k,p_detection,fine,p_pay_fine,bribe,share_stealer):
    p_detection=p_detection*w(share_stealer)
    return 1.0/(f_a(A,fine)+get_penalty_stealing(p_detection,fine,p_pay_fine,bribe))

    

def penalty(params):
    return 1


def f_a(A,fine):
    return (1+A)*fine


def set_game_custom(funct,x):
    #check that the input matches the equations
    var('w_aa,w_ab,w_ba,w_bb')
    for v in x:
        var('{}'.format(v))
    #make a loop  
    w_aa=funct[0][0]
    w_ab=funct[0][1]
    w_ba=funct[1][0]
    w_bb=funct[1][1]
    return[[w_aa,w_ab],[w_ba,w_bb]]
    #return [[Eq(w_aa,funct[0][0]),Eq(w_ab,funct[0][1])],[Eq(w_ba,funct[1][0]),Eq(w_bb,funct[1][1])]]

#don't like this very much
def set_param_values(game,param,value):
    var('{}'.format(param))
    for row in range(len(game)):
        for entry in range(len(game[row])):
            #var(param)
            game[row][entry]=game[row][entry].subs(param,value)
    return game

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

#shares
#strategies
def replicator(s,w,i):
    fitness_i = w[i]
    mean_fitness = np.dot(s,w)
    new_share = s[i]*np.divide(fitness_i,mean_fitness)
    return new_share

def discrete_replicator(s,w,i):
    return np.dot(s[i],(strategy_fitness_at_x(w,i,s)/average_fitness_at_x(w,s)))[0]

def get_delta_line_two(w,i,j):
    si,sj=symbols('si,sj')
    a = [si,sj]
    if j==0:
        a = [1-sj,sj]
    elif j==1:
        a = [si,1-si]
    wi = np.dot(a,w[i])
    wj = np.dot(a,w[j])
    z = wi-wj
    return [diff(z,si),diff(z,sj),[x if type(x)==Integer else 0 for x in z.args][0]]

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

def get_interior_equilibrium_two_by_two(w):
    delta_w12=get_delta_line_two(w,0,1)
    si,sj=symbols('si,sj')
    return solve(delta_w12[0]*si+delta_w12[1]*sj+delta_w12[2],si,sj)

def company_revenue(use,price,fine,rate_of_detection,share_stealers,p_pay_fine):
    return (1-share_stealers)*use*price + fine*rate_of_detection*share_stealers*p_pay_fine

def get_revenue_start_share(use,price,fine,rate_of_detection,cost):
    return (cost-use*price)/float(fine*rate_of_detection-use*price)
