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

#from Eichhorn_Distribution import GetTauC
from GetTauC import GetTauC, EstimateGammaD, A

PI = 3.14159265359

import time

config.initialize(gpu=False, slepc_args=['-mfn_ncv','18'])
#config.initialize(slepc_args=['-mfn_ncv','40'])
config.shell = True


from petsc4py.PETSc import Viewer
from petsc4py.PETSc import Vec

def NVz(i):
    return 0.5 * (sigmaz(i) - identity())


def Bis(rsP1s, rsNVs):
    NP1s = np.shape(rsP1s)[1]
    NNVs = np.shape(rsNVs)[1]
       
    B = np.zeros((NP1s, NNVs))
   
    for i in range(NNVs):
        for o in range(NP1s):
            dr = rsNVs[:,i] - rsP1s[:,o]
            r = np.linalg.norm(dr)
            C = dr[2] / r 
            B[o,i] = - A / (r**3) * (3*C**2-1)
    return B


def PlaceDefects(ppm, NDef, includeOrigin = False):
    a0 = 0.3567

    Ntot = 10*NDef
    dens = 8 * 10**(-6) * ppm / (a0**3)  
    L = (Ntot/dens)**(1.0/3.0)

    rs = np.zeros((3, Ntot))
    rs[0,includeOrigin:] = L * (np.random.rand(Ntot-includeOrigin) - 0.5)
    rs[1,includeOrigin:] = L * (np.random.rand(Ntot-includeOrigin) - 0.5)
    rs[2,includeOrigin:] = L * (np.random.rand(Ntot-includeOrigin) - 0.5)

    r2 = rs[0,:]**2 + rs[1,:]**2 + rs[2,:]**2
    k = np.argsort(r2)
    rs = rs[:,k[:NDef]]
    
    return rs

def evolutionOverACycle(s, vecT, tau, tauCs, tauPi,  eps, eps_tau, eps_stability, eps_spatial, Bs, Hdip):
    
    NP1s, NNVs = np.shape(Bs)
    
    OmegaRabi = (PI/2) / tauPi * (1+eps_stability)*(1+eps_spatial)
    Hrot = OmegaRabi * index_sum(sigmay())
    
    flip = np.array( [(-1)**(np.random.rand() > np.exp(-tau/tauCs[i])) for i in range(NP1s)] )
    for i in range(NNVs):
        Bs[:,i] = Bs[:,i] * flip
    
    H1 = Hdip + op_sum([np.sum(Bs[:,i])*NVz(i) for i in range(NNVs)])
    
    flip = np.array( [(-1)**(np.random.rand() > np.exp(-tauPi*(1+eps_tau+eps)/tauCs[i])) for i in range(NP1s)] )
    for i in range(NNVs):
        Bs[:,i] = Bs[:,i] * flip
    HMid =  Hrot + Hdip + op_sum([np.sum(Bs[:,i])*NVz(i) for i in range(NNVs)])

    flip = np.array( [(-1)**(np.random.rand() > np.exp(-tau/tauCs[i])) for i in range(NP1s)] )
    for i in range(NNVs):
        Bs[:,i] = Bs[:,i] * flip
    H2 = Hdip + op_sum([np.sum(Bs[:,i])*NVz(i) for i in range(NNVs)])

    H1.evolve(s, tau, result=vecT)
    HMid.evolve(vecT, tauPi*(1+eps_tau+eps), result = s)
    H2.evolve(s, tau, result=vecT)
    
    s = vecT.copy()
   
 
    return s,Bs
    

def DynamiteAvg(tau, tauPi, eps, ppmP1, ppmNV, K, cycles): 
    print("tau: ",tau)
    print("eps: ", eps)
    print("NP1sStart: ", NP1sStart)
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

        rsNVs = PlaceDefects(ppmNV, NNVs, includeOrigin = True)
        rsP1s = PlaceDefects(ppmP1, NP1sStart, includeOrigin = False)
        
        # Get the individual tauC from the distribution from Eichhorn et al paper
        #tauCs = [ GetTauC(np.random.rand(), tauCMean) for i in range(NP1s)]
        tauCs, to_remove = GetTauC(rsP1s, EstimateGammaD(ppmP1) )
        #print(tauCs)
        print(len(tauCs))
        print(np.min(tauCs))

        rsP1s = np.delete(rsP1s, to_remove, 1)
        NP1s = np.shape(rsP1s)[1]
        print("NP1s before/after: ", NP1sStart, " / ", NP1s)        
        
        Hdip = 0.0 * sigmax(0)

        #print(rsNVs[:,3])
        # Dipolar Piece        
        for i in range(NNVs):
            for o in range(i+1, NNVs):
                dr = rsNVs[:,i] - rsNVs[:,o]
                r = np.linalg.norm(dr)
                C = dr[2] / r 
                Jij =  A / r**3 * (3*C**2-1) 
                #print(i, o, Jij)

                Hdip += Jij * ( 0.25*(sigmax(i)*sigmax(o) + sigmay(i)*sigmay(o)) - NVz(i) * NVz(o) ) 
        #print(Hdip)
        #SpinFlip = index_product(sigmay())
        
        Bs = Bis(rsP1s, rsNVs) # Compute the interactions between the different fields

        # randomize initial fields
        flip = 0.5 * (-1)**(np.random.rand(NP1s) < 0.5) # factor of 0.5 accounts for the fact that the P1 is a spin 1/2 particle
        for l in range(L):
          Bs[:,l] = Bs[:,l] * flip

        for l in range(L):
            S[0, k, l] += real( s.dot(sigmay(l).dot(s,vecT)) )

        eps_tau =  0.25/40 *(0.5 - np.random.rand() )
        eps_spatial = 0.05 * np.random.randn()
        eps_stability = 0.025 * np.random.randn()

        print("eps_tau: ", eps_tau)
        print("eps_spatial: ", eps_spatial)
        print("eps_stability: ", eps_stability)
        #print("tauCs: ", tauCs)

        for i in range(1, cycles):
            s, Bs = evolutionOverACycle(s, vecT, tau, tauCs, tauPi, eps, eps_tau, eps_stability, eps_spatial, Bs, Hdip)
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
  if vals[0] == 'K' or vals[0] == "L" or vals[0]=="cycles" or vals[0] == "rep":
    params[vals[0]] = int(vals[1]) 
  elif vals[0] == "outDir":
    params[vals[0]] = (vals[1].split())[0]
  else:
    params[vals[0]] = float(vals[1])

outFile = params["outDir"] + "/Bauch_CMPG_"
for k in ["cycles", "K", "tau", "tauPi",  "eps", "ppmNV", "ppmP1", "rep"]:
  outFile += k + "_" + str(params[k]) + "__"
outFile += ".npy"
print("Saving to: ", outFile)

### PREPARE INITIAL STATE
L = params["L"]
config.L = L

NP1sStart = np.max([250, int(3*4*params["ppmP1"] / params["ppmNV"])])

data = DynamiteAvg(
  params["tau"],
  params["tauPi"],
  params["eps"],
  params["ppmP1"],
  params["ppmNV"]/4,
  params["K"],
  cycles = params["cycles"])


Data = {"data":data, "params":params}
np.save(outFile, Data)
print("DONE")
