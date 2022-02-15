import sys
import os
from sys import argv
from math import sin, sqrt, fabs

import numpy as np

from dynamite import config

from dynamite.states import State
from dynamite.operators import sigmax, sigmay, sigmaz, identity, index_sum, index_product, op_sum
from dynamite.computations import entanglement_entropy

from numpy import ceil, abs, real
from numpy import genfromtxt
from numpy import linspace
from numpy.random import randn

from math import sin

PI = 3.14159265359

import time

config.initialize(gpu=False, slepc_args=['-mfn_ncv','18'])
#config.initialize(slepc_args=['-mfn_ncv','40'])
config.shell = True


from petsc4py.PETSc import Viewer
from petsc4py.PETSc import Vec



def Bis(rsP1s, rsNVs):
    NP1s = np.shape(rsP1s)[1]
    NNVs = np.shape(rsNVs)[1]
    
    B = np.zeros((NP1s, NNVs))

    for i in range(NNVs):
        for o in range(NP1s):
            dr = rsNVs[:,i] - rsP1s[:,o]
            r = np.linalg.norm(dr)
            C = dr[2] / r 
            B[o,i] = (2*np.pi) * 52 / (r**3) * (1 - 3*C**2)
    return B


def PlaceNVsAndP1s(NVs, ppmP1, ppmNV):
    a0 = 0.3567

    NP1s = int(np.round(NVs * 2*ppmP1 / ppmNV))
    
    
    if ppmP1 > 0:
        NtotP1s = 10*NP1s
        dens = 8 * 10**(-6) * ppmP1 / (a0**3)  
        L = (NtotP1s/dens)**(1.0/3.0)

#         print(L)


        rsP1s = np.zeros((3, NtotP1s))
        rsP1s[0,:] = L * (np.random.rand(NtotP1s) - 0.5)
        rsP1s[1,:] = L * (np.random.rand(NtotP1s) - 0.5)
        rsP1s[2,:] = L * (np.random.rand(NtotP1s) - 0.5)

        rP1s = rsP1s[0,:]**2 + rsP1s[1,:]**2 + rsP1s[2,:]**2
        kP1s = np.argsort(rP1s)
        rsP1s = rsP1s[:,kP1s[:NP1s]]
        
    else:
        rsP1s = np.zeros((3,0))
    
    
    NtotNVs = 10*NVs
    dens = 8 * 10**(-6) * ppmNV / (a0**3)  
    L = (NtotNVs/dens)**(1.0/3.0)
    
#     print(L)
    
    rsNVs = np.zeros((3, NtotNVs))
    rsNVs[0,1:] = L * (np.random.rand(NtotNVs-1) - 0.5)
    rsNVs[1,1:] = L * (np.random.rand(NtotNVs-1) - 0.5)
    rsNVs[2,1:] = L * (np.random.rand(NtotNVs-1) - 0.5)
          
    rNVs = rsNVs[0,:]**2 + rsNVs[1,:]**2 + rsNVs[2,:]**2
    kNVs = np.argsort(rNVs)
    rsNVs = rsNVs[:,kNVs[:NVs]]
    
    
    
    return rsNVs, rsP1s

def evolutionOverACycle(s, vecT, tau, tauC, tauPi,  eps, Bs, Hdip):
    
    NP1s, NNVs = np.shape(Bs)
    
    OmegaRabi = (PI/2) / tauPi
    Hrot = OmegaRabi * index_sum(sigmay())
    
    flip = (-1)**(np.random.rand(NP1s) < tau/tauC)
    for i in range(NNVs):
        Bs[:,i] = Bs[:,i] * flip
    
    H1 = Hdip + op_sum([np.sum(Bs[:,i])*(0.5*identity() - 0.5*sigmaz(i)) for i in range(NNVs)])
    
    flip = (-1)**(np.random.rand(NP1s) < tauPi*(1+eps)/tauC)
    for i in range(NNVs):
        Bs[:,i] = Bs[:,i] * flip
    HMid =  Hrot + Hdip + op_sum([np.sum(Bs[:,i])*(0.5*identity() - 0.5*sigmaz(i)) for i in range(NNVs)])

    flip = (-1)**(np.random.rand(NP1s) < tau/tauC)
    for i in range(NNVs):
        Bs[:,i] = Bs[:,i] * flip
    H2 = Hdip + op_sum([np.sum(Bs[:,i])*(0.5*identity() - 0.5*sigmaz(i)) for i in range(NNVs)])

    H1.evolve(s, tau, result=vecT)
    HMid.evolve(vecT, tauPi*(1+eps), result = s)
    H2.evolve(s, tau, result=vecT)
    
    s = vecT.copy()
   
 
    return s,Bs
    

def DynamiteAvg(tau, tauC, tauPi, eps, ppmP1, ppmNV, K, cycles): 
    print("tau: ",tau)
    print("tauC: ",tauC)
    print("eps: ", eps)
    S = np.zeros( (cycles, K, L) )
    
    print("K = ", K)
    for k in range(K):
        print(k ,", ", end = "")

        NNVs = L

        s = State(state="U"*L)
        vecT = s.copy()

        # Need to rotate into the Y basis to prepare the right initial state
        for i in range(L):
            Hrot = sigmax(i)
            Hrot.evolve(s,-PI/4,result=vecT)
            s = vecT.copy()

        rsNVs, rsP1s = PlaceNVsAndP1s(NNVs, ppmP1, ppmNV)
        NP1s = np.shape(rsP1s)[1]
        #print(np.shape(rsP1s))
#         print(rsNVs)
        Hdip = 0.0 * sigmax(0)
        #print(rsNVs[:,3])
        # Dipolar Piece
        for i in range(NNVs):
            for o in range(i+1, NNVs):
                dr = rsNVs[:,i] - rsNVs[:,o]
                r = np.linalg.norm(dr)
                C = dr[2] / r 
                Jij =  2*np.pi * 52 / r**3 * (1-3*C**2) 
                #print(i, o, Jij)

                Hdip += Jij * ( 0.25*(sigmax(i)*sigmax(o) + sigmay(i)*sigmay(o)) - (0.5*identity() - 0.5*sigmaz(i)) * (0.5*identity() - 0.5*sigmaz(o)) ) 
        #print(Hdip)
        SpinFlip = index_product(sigmay())

        
        Bs = Bis(rsP1s, rsNVs) # Compute the interactions between the different fields

        # randomize initial fields
        flip = 0.5 * (-1)**(np.random.rand(NP1s) < 0.5) # factor of 0.5 accounts for the fact that the P1 is a spin 1/2 particle
        for l in range(L):
          Bs[:,l] = Bs[:,l] * flip

        for l in range(L):
            S[0, k, l] += real( s.dot(sigmay(l).dot(s,vecT)) )


        for i in range(1, cycles):
            s, Bs = evolutionOverACycle(s, vecT, tau, tauC, tauPi, eps, Bs, Hdip)
            for l in range(L):
        #         print(s.dot(sigmay(l).dot(s,vecT)))
                S[i, k,l] += real( s.dot(sigmay(l).dot(s,vecT)) ) 

    return S




# Load Variables
opts_file = sys.argv[1]
opts = open(opts_file, 'r')

params = {}

for line in opts:
  vals = line.split(",")
  if vals[0] == 'K' or vals[0] == "L" or vals[0]=="cycles":
    params[vals[0]] = int(vals[1]) 
  elif vals[0] == "outDir":
    params[vals[0]] = (vals[1].split())[0]
  else:
    params[vals[0]] = float(vals[1])

outFile = params["outDir"] + "/CMPG_"
for k in ["cycles", "K", "tau", "tauPi",  "eps", "tauC", "ppmNV", "ppmP1"]:
  outFile += k + "_" + str(params[k]) + "__"
outFile += ".npy"
print("Saving to: ", outFile)

### PREPARE INITIAL STATE
L = params["L"]
config.L = L

data = DynamiteAvg(
  params["tau"],
  params["tauC"],
  params["tauPi"],
  params["eps"],
  params["ppmP1"],
  params["ppmNV"]/4,
  params["K"],
  cycles = params["cycles"])


Data = {"data":data, "params":params}
np.save(outFile, Data)
print("DONE")
