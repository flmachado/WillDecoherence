import numpy as np

hbar = 1.05e-34
mu0 = 4*np.pi * 1e-7
mue = 1.76e11
A = 2*np.pi*(mu0/(4*np.pi)) * hbar/(2*np.pi) * mue**2 * 1e27 * 1e-6 # To work in units of MHz and nm

def Cij(rsP1s, i, j):
    dr = rsP1s[:,i] - rsP1s[:,j]
    r = np.linalg.norm(dr)
    nz = dr[2]/r
    
    return A/r**3 * (1-2*nz**2), A/r**3 * 0.5*(1-0.25*(1-nz**2) )


def EstimateGammaD(ppm):
    FittedCoeff = 0.061
    return FittedCoeff*ppm

def GetTauC(rsP1s, GammaD):
    NP1s = np.shape(rsP1s)[1]
    randomState = 0.5 * (-1)**(np.random.rand(NP1s) < 0.5)

    # Compute the local Magnetic field from a random state to estimate the tau_c
    Bis = np.zeros(NP1s)
    for i in range(NP1s): 
      Bis[i] += np.random.choice([-120, -90, 0, 90, 120], p=[1.0/12, 1.0/4, 1.0/3, 1.0/4, 1.0/12])
      
    Omega = np.zeros((NP1s,NP1s) )
    for i in range(NP1s):
      for j in range(i+1, NP1s):
        Cpar, Cperp = Cij(rsP1s, i,j)
        Bis[i] += Cpar * randomState[j]
        Bis[j] += Cpar * randomState[i]
        Omega[i,j] = Cperp            

    Rates = np.zeros((NP1s, NP1s))
    to_remove = []
    for i in range(NP1s):  
      for j in range(i+1,NP1s):
        if  Omega[i,j] > np.sqrt( (Bis[i]-Bis[j])**2 + GammaD**2):
          #don't consider these two spins anymore because they are too strongly coupled
          to_remove.append(i)
          to_remove.append(j)
          continue
        Rates[i,j] = Omega[i,j]**2 * GammaD/ (GammaD**2+(Bis[i]-Bis[j])**2)
        Rates[j,i] = Omega[i,j]**2 * GammaD/ (GammaD**2+(Bis[i]-Bis[j])**2)

    to_remove = list(set(to_remove))
    Rates = np.delete(Rates, to_remove, 0)
    Rates = np.delete(Rates, to_remove, 1)
    tauCs = 1.0 / np.sum(Rates, axis = 1)
    return tauCs, to_remove

