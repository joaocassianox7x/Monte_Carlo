#PARALLEL MONTE-CARLO FOR HARMONIC OSCILATOR

import numpy as np
from multiprocessing import Pool
import time
import matplotlib.pyplot as plt


def monte_carlo_pi(n):
    np.random.seed()
    return(4*np.sum((np.random.rand(n)**2+np.random.rand(n)**2)<=1)/n)

def funcao_chute(x,alpha=0.5):
    return(np.exp(-alpha*x**2))

def energia(x,alpha=0.5):
    return(alpha+(1/2-2*alpha**2)*x**2)

def metropolis(n):
    np.random.seed()
    x0 = 0.5
    xv = x0
    E_s = []
    for i in range(n):
        xn = xv+(np.random.rand()-0.5)*0.5 
        if funcao_chute(xn)**2/funcao_chute(xv)**2>=np.random.rand():
            xi=xn
            xv=xn
        else:
            xi=xv
        E_s.append(energia(xi))        
    return np.sum(E_s)/n

#parâmentros
M = int(1e7) # numero de amostras
M_per_thread = int(M/N_threads) #numero de amostra para cada processo
L_per_thread = [M_per_thread]*N_threads #lista com número de amostra para cada processo
t=[]

for N_threads in range(1,8+1):
    t0 = time.time()
    pool = Pool(processes=N_threads) #abrindo o ambiente para o paralelismo
    lista_E_ns = pool.map(metropolis,L_per_thread) #lista com os valores de Pi para cada um dos processos
    t.append(time.time()-t0)
    
plt.plot(range(1,len(t)+1),t,"-o",lw=2.5)
plt.xlabel("Numero de Processos")
plt.ylabel("Tempo [s]")
plt.title("N = "+str(M))
plt.xlim(1,8)
plt.grid(True)
plt.tight_layout()
plt.savefig("figura.png")