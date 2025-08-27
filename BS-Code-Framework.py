# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 10:28:32 2025

@author: Human
"""
#%%
import numpy as np 
import matplotlib.pyplot as plt
from math import factorial
from scipy.stats import norm

r = 0.05
sigma = 0.25
T = 1

# FUNCTIONS FOR EUROPEAN PUTS AND CALLS    

def EPut(X, S0, sigma, T, r): 
    d1 = (1/(sigma*np.sqrt(T)))*(np.log(S0/X) + (r + sigma**2/2)*T)
    d2 = d1 - sigma*np.sqrt(T)
    
    p = norm.cdf(-d2)*X*np.exp(-r*T) - norm.cdf(-d1)*S0
    gamma = (norm.pdf(d1))/(S0*sigma*np.sqrt(T))
    delta = norm.cdf(d1)-1
    #print("Gamma = ", gamma)
    #print("Delta = ", delta)
    return(np.array([p,delta,gamma]))

def ECall(X, S0, sigma, T, r): 
    d1 = (1/(sigma*np.sqrt(T)))*(np.log(S0/X) + (r + sigma**2/2)*T)
    d2 = d1 - sigma*np.sqrt(T)
    c = S0*norm.cdf(d1) - X*np.exp(-r*T)*norm.cdf(d2)
    
    gamma = (norm.pdf(d1))/(S0*sigma*np.sqrt(T))
    delta = norm.cdf(d1)
    return(np.array([c,delta,gamma]))


"""
Plot of option price dynamics
"""

#Strikes vs C
strikes = np.arange(55,200, 0.1)
S = [90,100,110, 150]
for s in S: 
     calls = ECall(strikes,s,sigma,T,r)[0]
     plt.plot(strikes, calls, label = "S = " + str(s))

plt.legend()
plt.ylabel("Option Price (C)")
plt.xlabel("Strike Price (X)")
plt.title("Strike Price Vs Option Price (European Call) with T=1")     
plt.show()

#Spots vs C
spots = np.arange(50,200, 0.1)
X = [90,100,110, 150]
for x in X: 
     calls = ECall(x,spots,sigma,T,r)[0]
     plt.plot(spots, calls, label = "X = " + str(x))

plt.legend()
plt.ylabel("Option Price (C)")
plt.xlabel("Spot Price (S)")
plt.title("Spot Price Vs Option Price (European Call) with T=1")     
plt.show()

#T vs. C     
times = np.arange(0,10,0.005)
for x in X: 
    calls = ECall(x,100,sigma,times,r)[0]
    plt.plot(times, calls, label = "X = " + str(x))

plt.legend()
plt.ylabel("Option Price (C)")
plt.xlabel("Time to Maturity (T)")
plt.title("Time to Maturity Vs Option Price (European Call) with S=100")     
plt.show()

for s in S: 
    calls = ECall(100,s,sigma,times,r)[0]
    plt.plot(times, calls, label = "S = " + str(s))

plt.legend()
plt.ylabel("Option Price (C)")
plt.xlabel("Time to Maturity (T)")
plt.title("Time to Maturity Vs Option Price (European Call) with X=100")     
plt.show()

"""
GREEKS
"""
##DELTA##

#T vs Delta
for s in S: 
    delta = ECall(100,s,sigma,times,r)[1]
    plt.plot(times, delta, label = "S = " + str(s))

plt.legend()
plt.ylabel("Delta")
plt.xlabel("Time to Maturity (T)")
plt.title("Time to Maturity Vs Delta (European Call) with X=100")     
plt.show()    

#S vs Delta   
for x in X: 
    delta  = ECall(x,spots,sigma,T,r)[1]
    plt.plot(spots, delta, label = "X = " + str(x))
    
plt.legend()
plt.ylabel("Delta")
plt.xlabel(" Spot Price (S)")
plt.title("Spot Price Vs Delta (European Call)")     
plt.show() 

#X vs Delta
for s in S: 
    delta  = ECall(strikes,s,sigma,T,r)[1]
    plt.plot(strikes, delta, label = "S = " + str(s))
    
plt.legend()
plt.ylabel("Delta")
plt.xlabel(" Strike Price (X)")
plt.title("Strike Price Vs Delta (European Call)")     
plt.show() 

##GAMMA##

#T vs gamma
times = np.arange(0.2,10,0.1)
for s in S: 
    delta = ECall(100,s,sigma,times,r)[2]
    plt.plot(times, delta, label = "S = " + str(s))

plt.legend()
plt.ylabel("Gamma")
plt.xlabel("Time to Maturity (T)")
plt.title("Time to Maturity Vs Gamma (European Call) with X=100")     
plt.show()    

#S vs Delta   
spots = np.arange(40,250, 0.1)
for x in X: 
    delta  = ECall(x,spots,sigma,T,r)[2]
    plt.plot(spots, delta, label = "X = " + str(x))
    
plt.legend()
plt.ylabel("Gamma")
plt.xlabel(" Spot Price (S)")
plt.title("Spot Price Vs Gamma (European Call)")     
plt.show() 

#X vs Delta
strikes = np.arange(50,250, 0.1)
for s in S: 
    delta  = ECall(strikes,s,sigma,T,r)[2]
    plt.plot(strikes, delta, label = "S = " + str(s))
    plt.legend()
plt.ylabel("Gamma")
plt.xlabel(" Strike Price (X)")
plt.title("Strike Price Vs Gamma (European Call)")     
plt.show() 


"""
Zero-cost Collar Code and Plots
"""


#Sell Call with X = 120
#Buy Put with X = 100

# Xp<Xc
Xc = 120
Xp = 80
s0 = 100
p1 = EPut(Xp, s0, sigma, T, r)[0]
c1 = ECall(Xc, s0, sigma, T, r)[0]

spots = np.arange(Xp-20, Xc + 20, 0.01)
puts = [max(0,Xp-spots[i])-p1 for i in range(0, len(spots))]
calls1 = [c1 + min(0,Xc-spots[i]) for i in range(0, len(spots))]
payout = [max(0,Xp-spots[i]) - max(0, spots[i]-Xc) for i in range(0, len(spots))]
profit = [spots[i] + payout[i] - s0 for i in range(0, len(spots))]

plt.plot(spots, puts, label = "Long Put", ls = "--")
plt.plot(spots, calls1, label = "Short Call", ls = "--")
plt.plot(spots, payout, label= "Payout", ls = "--")
plt.plot(spots, profit, label = "Profit")
plt.legend()
plt.grid(True)
plt.title("Spot Price Vs Option Values (Xp < Xc)")
plt.ylabel("Option Value")
plt.xlabel("Spot Price (S)")
plt.show()



# Xp>Xc
Xc = 80
Xp = 120
s0 = 100
p1 = EPut(Xp, s0, sigma, T, r)[0]
c1 = ECall(Xc, s0, sigma, T, r)[0]

spots = np.arange(Xc-20, Xp + 40, 0.005)
puts = [max(0,Xp-spots[i])-p1 for i in range(0, len(spots))]
calls1 = [c1 + min(0,Xc-spots[i]) for i in range(0, len(spots))]
payout = [max(0,Xp-spots[i]) - max(0, spots[i]-Xc) for i in range(0, len(spots))]
profit = [spots[i] + payout[i] - s0 for i in range(0, len(spots))]

plt.plot(spots, puts, label = "Long Put", ls = "--")
plt.plot(spots, calls1, label = "Short Call", ls = "--")
plt.plot(spots, payout, label= "Payout", ls = "--")
plt.plot(spots, profit, label = "Profit")
plt.legend()
plt.grid(True)
plt.title("Spot Price Vs Option Values (Xp > Xc)")
plt.ylabel("Option Value")
plt.xlabel("Spot Price (S)")
plt.show()
# Keep s0= 100. Adjust Xc and Xp
Xp = np.arange(70,130, 10)
Xc = np.arange(130,70, -10)

#Payoff (No underlying owned)
for j in range(0,len(Xp)): 
    p1 = EPut(Xp[j], s0, sigma, T, r)[0]
    c1 = ECall(Xc[j], s0, sigma, T, r)[0]
    spots = np.arange(60, 140, 1)
    puts = [max(Xp[j] - spots[i], 0)-p1 for i in range(0, len(spots))]
    calls = [c1 + min(0,Xc[j]-spots[i]) for i in range(0, len(spots))]
    payout = [max(0,Xp[j]-spots[i]) - max(0, spots[i]-Xc[j]) for i in range(0, len(spots))]
    profit = [spots[i] + payout[i] - s0 for i in range(0, len(spots))]
    
    plt.plot(spots,payout)

plt.title("Spot Price Vs Payoff of Strategy Xp < Xc")    
plt.ylabel("Payout (Profit)")
plt.xlabel("Spot Price S")
plt.grid(True)
plt.legend()
plt.show()


#Profit (underlying owned)
Xp = np.arange(80,120,10)
Xc = np.arange(100,140, 10)
for j in range(0,len(Xp)): 
    p1 = EPut(Xp[j], s0, sigma, T, r)
    c1 = ECall(Xc[j], s0, sigma, T,r)
    spots = np.arange(60, 140, 1)
    puts = [max(Xp[j] - spots[i], 0)-p1 for i in range(0, len(spots))]
    calls = [c1 + min(0,Xc[j]-spots[i]) for i in range(0, len(spots))]
    payout = [max(0,Xp[j]-spots[i]) - max(0, spots[i]-Xc[j]) for i in range(0, len(spots))]
    profit = [spots[i] + payout[i] - s0 for i in range(0, len(spots))]
    
    plt.plot(spots, profit)

plt.title("Spot Price Vs Profit of Strategy Xp < Xc")    
plt.ylabel("Profit")
plt.xlabel("Spot Price S")
plt.grid(True)
plt.legend()
plt.show()

#Rolling Collar
Xp = [90,95,105,110,130,145]
Xc = [110,120,130,135,150,170]
S0 = np.arange(100, 160,10)
k=0
for j in S0: 
    p1 = EPut(Xp[k], j, sigma, T, r)[0]   
    c1 = ECall(Xc[k], j, sigma, T, r)[0]
    spots = np.arange(60, 200, 1)
    puts = [max(Xp[k] - spots[i], 0)-p1 for i in range(0, len(spots))]
    calls = [c1 + min(0,Xc[k]-spots[i]) for i in range(0, len(spots))]
    payout = [max(0,Xp[k]-spots[i]) - max(0, spots[i]-Xc[k]) for i in range(0, len(spots))]
    profit = [spots[i] + payout[i] - j for i in range(0, len(spots))]
    plt.plot(spots, profit)
    k= k+1

plt.title("Rolling Collar")    
plt.ylabel("Profit")
plt.xlabel("Spot Price S")
plt.grid(True)
plt.legend()
plt.show()

#Showing value of the options combined vs Spot 
spots = np.arange(50, 150, 1)
X = [70,80,100,120]
for x in X:
    p = EPut(x, spots, sigma, T, r)[0]   
    c = ECall(x+10, spots, sigma, T, r)[0]
    total = c-p
    plt.plot(spots,total, label = "Xp = " + str(x)+ ", Xc " + str(x+10))
    
plt.ylabel("Option Prices (Call-Put)")
plt.xlabel("Spot Prices")
plt.title("Spot Prices vs Option Prices (C-P) for a collar")
plt.grid(True)
plt.legend()
plt.show()
    
#Showing the cost of the options vs spot prices   
spots = np.arange(50, 150, 1)
X = [70,80,100,120]
for x in X:
    p = EPut(x-10, spots, sigma, T, r)[0]   
    c = ECall(x+40, spots, sigma, T, r)[0]
    total = c-p
    plt.plot(spots,total, label = "Xp = " + str(x-10)+ ", Xc " + str(x+40))
    
plt.ylabel("Option Prices (Call-Put)")
plt.xlabel("Spot Prices")
plt.title("Spot Prices vs Option Prices (C-P) for a Collar")
plt.grid(True)
plt.legend()
plt.show()







