# made with python version 3.9.16
# pip install pandas
# pip install matplotlib
# pip install scipy
# pip install BSpline 
# pip install openpyxl
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from scipy import stats
from scipy import linalg

#os.chdir('C://Users//mzgra//OneDrive//Dissertation Research//Mike Grabel//Data and Code')
os.chdir(r'C:\Users\mzgra\OneDrive\Dissertation Research\BARS Python')

data = pd.read_excel('NormalData.xlsx', header = None)

x = data[0] # main variable to perform splines on
y = data[1]
#plt.plot(x, y)

n = len(x)



# normalize x
def normalize(x):
    nx = (x - min(x))/(max(x)-min(x))
    return nx

nx = normalize(x)

# parameers


# initialize knots  
def initialknots(k):
    knots = np.zeros(k)
    for i in range(k):
        knots[i] = (i+1)/(k+1)    
    
    return knots



# plt.plot(nx,y)

# for i in knots:
#     plt.axvline(x=i, ls = '--', color = 'grey')
    
# plt.show()

def SetRJprobs(c, k, maxknots):
# Set probabilities of adding or removing a knot given the number of knots
    birthprob = np.zeros(maxknots) 
    deathprob = np.zeros(maxknots)
    for k in range(maxknots):
        if k == (maxknots-1): # if number of knots = maxknots prob of adding a new knot is 0
            birthprob[k] = 0
        else:
            a = np.random.uniform(0,(k+1)+1) # calculate the ratio P(k+1)/P(k) by sampling from uniform prior for the probability of generating a new knot at this number of knots
            b = np.random.uniform(0,(k+1))
            birthprob[k] = c * min(1, a/b) # multiply by parameter c which controls the probability
        if k == 0:  # if number of knots = 1 prob of removing a knot is 0
            deathprob[k] = 0
        else:
            b = np.random.uniform(0,(k+1)) # calculate the ratio P(k-1)/P(k)
            d = np.random.uniform(0,(k+1)-1)
            deathprob[k] = c * min(1, d/b) 
            
    return birthprob, deathprob            
        


def fitnormalmodel(x, y, k, knots_ext, n):
    flag = False
    X = BSpline.design_matrix(x, knots_ext, 3)#, extrapolate=True) # fit spline basis at knot points
    X = X[:,2:]
    aX = X.toarray()
    aXt = np.transpose(aX) # get X'
    
    Xty = np.dot(aXt, y) # X'Y
    XtX = np.dot(aXt, aX) # X'X
    cond_num = np.linalg.cond(XtX)
    if cond_num > 1 / np.finfo(float).eps:
        flag = True
        return 0,0,0,0, np.inf, flag
    B = linalg.solve(XtX, Xty) # solve (X'X)B = X'Y for B
    
    mu = np.dot(aX, B) # fit model XB = Y
    
    LL = np.sum(stats.norm.logpdf(y, mu)) # get log likelihood of the fit of this model 
    
    bic = -2.0 * LL + k * np.log(n)
    
    return aX, B, mu, LL, bic, flag

def checkcand(knot_cand, knots_ext):
   
    norm = np.zeros(len(knots_ext))
    for i in range(len(knots_ext)):
        norm[i] = (np.abs(knot_cand - knots_ext[i])< 2e-4)
    if np.any(norm == 1):
        return 1
    elif np.all(norm == 0):
        return 0

def getdensity(knot_cand, knots, tau): 
    d = np.zeros(len(knots))              
    for i in range(len(knots)):
        d[i] = stats.beta.pdf(knot_cand, tau*knots[i], (1 - knots[i])* tau)
    D = sum(d)
    return D

def getdensities(knot_cand, s, tau):
    dens1 = stats.beta.pdf(knot_cand, tau*s, tau*(1-s))
    dens2 = stats.beta.pdf(s, tau*s, tau*(1-knot_cand))
    return dens1, dens2

def BARS(nx, y):
    tau = 50
    c = 0.4
    maxknots = 60
    # min_knots = 1
    k = 25
    num_iter = 10000
    # threshold = 0.01
    knots = initialknots(k)
    n = len(nx)
    # add 0 and 1 as boundary knots
    knots_ext = np.r_[0,0,0,0, knots, 1,1,1,1]
    [aX, B, m, LL_curr, bic, flag] = fitnormalmodel(nx, y, k, knots_ext, n)
    best_bic = np.inf
    
    # while len(knots) > min_knots:
    # idx_remove = []
    
    # for i in range(len(knots)):
    #     knotsn = knots[knots != knots[i]]
    #     knotsn_ext = np.r_[0,0,0,0, knotsn, 1,1,1,1]
    #     [_, _, _, LL_cand, _, _] = fitnormalmodel(nx, y, k, knotsn_ext, n)
    #     deltaLL = LL_full - LL_cand
    #     if deltaLL < threshold:
    #         LL_full = LL_cand
    #         idx_remove.append(i)
    
    # # if not idx_remove:
    # #     break
            
    # knots = np.delete(knots, idx_remove)
    # k = k - len(idx_remove)

    # knots_ext = np.r_[0,0,0,0, knots, 1,1,1,1]
    # [aX, B, m, LL_curr, bic, flag] = fitnormalmodel(nx, y, k, knots_ext, n)
    [birthprob, deathprob] = SetRJprobs(c,k, maxknots)

    for i in range(num_iter):
        print(i)
        u = np.random.uniform(0,1)  # sample from uniform(0,1)
        if u < birthprob[k]: # if that sample is less than the birth probability at the current number of knots
            cc = 1    
            while cc == 1:
                # birth step build model 
                s = np.random.choice(knots[knots != 0]) # randomly select a current knot 
                    
                # center s around a beta dist 
                alpha = s*tau  #alpha and beta are multiplied by parameter tau which controls the spread for the candidate knots
                beta = (1-s)*tau
                t = np.random.beta(alpha, beta, 1000) # generate a beta distrbution centered around the sampled knot 
                    
                knot_cand = np.random.choice(t) # randomly select a candidate knot from the distribution
                cc = checkcand(knot_cand, knots_ext)   
                
            xi_cand = np.sort(np.r_[knots, knot_cand]) # append it to the current set of knots
            k_cand = k + 1 # add 1 to the number of knots
            # create model with candidate knots
            xi_ext = np.r_[0,0,0,0, xi_cand, 1,1,1,1]
            [aX, B, mu, LL_cand, bic, flag] = fitnormalmodel(nx, y, k_cand, xi_ext, n) # fit a normal model given the candidate set of knots 
            if flag == True:
                i -= 1
                continue
            else:
                D = getdensity(knot_cand, knots, tau) # get densities centered at the candidate knot
                acceptance_prob = np.exp(LL_cand - LL_curr + np.log(k) - np.log(D) - 0.5 * np.log(n)) # acceptance probability of this model
                
        elif (1.0-u) < deathprob[k]: # if 1 - the random sample from uniform is less than the death probability at the current number of knots 
            # death step build model
            s = np.random.choice(knots[knots != 0]) # randomly select a current knot
            xi_cand = np.sort(knots[knots != s]) # remove it from the set of knots
            k_cand = k - 1 # subtract 1 from the number of knots
            xi_ext = np.r_[0,0,0,0, xi_cand, 1,1,1,1]
            [aX, B, mu, LL_cand, bic, flag] = fitnormalmodel(nx, y, k_cand, xi_ext, n) # fit a normal model given the candidate set of knots 
            if flag == True:
                i -= 1
                continue
            else:
                D = getdensity(s, xi_cand, tau) # get densities centered at the candidate knot
                acceptance_prob = np.exp(LL_cand - LL_curr + np.log(k_cand) - np.log(D) - 0.5 * np.log(n)) # acceptance probaility of this model
        else: # else 1 - birth - death relocate a knot
            #relocate step 
            cc = 1    
            while cc == 1:
                    
                s = np.random.choice(knots[knots != 0]) # randomly select a current knot
                    
                # center s around a beta dist 
                alpha = s*tau
                beta = (1-s)*tau
                t = np.random.beta(alpha, beta, 1000)
                knot_cand = np.random.choice(t) # randomly select from the beta distrbution centered at the sample knot
                cc = checkcand(knot_cand, knots_ext)
                
            xim = knots[knots!=s] # remove this knot
            xi_cand = np.sort(np.r_[xim, knot_cand]) # and add the new knot
            k_cand = k # number of knots remains the same 
            xi_ext = np.r_[0,0,0,0, xi_cand, 1,1,1,1]
            [aX, B, mu, LL_cand, bic, flag] = fitnormalmodel(nx, y, k_cand, xi_ext, n) # fit a normal model given the candidate set of knots 
            if flag == True:
                i -= 1
                continue
            else:
                D1, D2 = getdensities(knot_cand, s, tau) # get densities 
                acceptance_prob = np.exp(LL_cand - LL_curr + np.log(D1) - np.log(D2)) # get acceptance probability
        u = np.random.uniform(0,1) # randomly select from the uniform(0,1)
        if u < acceptance_prob: # if it is less than the acceptance probability for a given model above switch out the models 
            # temp = curr
            # curr = cand
            # cand = temp
            LL_curr = LL_cand
            knots = xi_cand
            k = k_cand
            aX_curr = aX
            m = mu # return the fit XB
            B_curr = B
        else:
            aX_curr = aX
            B_curr = B
        if bic < best_bic:
            best_bic = bic    
            best_LL = LL_curr
            best_knots = knots
            best_k = k
            best_aX = aX_curr
            best_mu = m
            best_B = B_curr
            
    return best_mu, best_aX, best_B, best_knots, best_k, best_LL


m, X, B, knots, k, LL_curr = BARS(nx, y) # returns XB

plt.plot(nx, y)#, color = 'navy')
plt.plot(nx, m, color = 'red')
plt.vlines(x = knots, ymin = -2, ymax = 2, color = 'gray', linestyles='dashed')
    

def unitprior(B, J, n, p):
    log_prior = 0.0
    for i in range(1, p):
        log_prior += -0.5 * (B[i]**2) / n
    return log_prior    
    
def RandBeta(X, y, B, LL_curr, knots):
    
    MHT = -10    
    iters = 0
    B_curr = B
    p = np.shape(X)[1]
    n = np.shape(X)[0]
    for i in range(3):
        iters += 1
        Xt = np.transpose(X)
        XtX = np.dot(Xt, X)
        J = linalg.cholesky(XtX)
        z = np.random.multivariate_normal(mean = np.zeros(p), cov = np.identity(p)) # ~ N(0,I)
        A = linalg.solve(J, z) # solve J'A = z for A
        B_cand = B + A # new cand beta 
        mu_cand = np.dot(X, B_cand) # fit model XB = Y
        # log likelihood
        LL_cand = np.sum(stats.norm.logpdf(y, mu_cand)) # get log likelihood of the fit of this model 
        # priors
        curgi = unitprior(B_curr, J, n, p)
        lastgi = unitprior(B_cand, J, n, p)
        # normal proposal density priors
        XtWX = np.dot(np.dot(Xt, np.diag(mu_cand)), X)
        curhi = sum(np.random.multivariate_normal(mean = B_curr, cov = XtWX))
        lasthi = sum(np.random.multivariate_normal(mean = B_cand, cov = XtWX))
        
        r = (LL_cand - LL_curr) + (curgi - lastgi) - (curhi - lasthi)

        if r > MHT:
            i += 1
            u = r - 1.0
        else:
            u = np.random.uniform(0,1)
            u = np.log(u)
        if u < r:
            B_curr = B_cand
            LL_curr = LL_cand
    return(B_curr)

# B_fin = RandBeta(X, y, B, LL_curr, knots)
# B_fin