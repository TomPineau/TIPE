import numpy as np
import matplotlib.pyplot as plt

#En 1 dimension

def valuation(s):
    i=0
    while s%2==0:
        i+=1
        s=s//2
    return i

def Suppression1D(L,epsilon):
    T=L[:]
    for k in range(len(T)):
        if abs(T[k])<epsilon:
            T[k]=0
    return T

def Zeros(L):
    i=0
    for k in range(len(L)):
        if L[k]==0:
            i+=1
    return i

def ZerosHaar1D(L) :
    return Zeros(Suppression1D(Haar1D(L),epsilon))

def Pourcentage_suppression(L) :
    print(100*ZerosHaar1D(f(L))/len(L),'%')

def f(x) :
    return (2.5*(0.25+x)*(1.5-x)+0.1*(0.5-x)*np.sin(40*x))+0.03*np.cos(250*x)*np.exp(np.sin(10*(x**2)))

def g(x) :
    return 2.5*(0.25+x)*(1.5-x)+0.1*(0.5-x)*np.sin(40*x)

#Décomposition en ondelettes de Haar

def Haar1D(L):
    n=len(L)
    A=L[:]
    C=[]
    for i in range(valuation(n)):
        a=[]
        c=[]
        for j in range(0,len(A)//2):
            a.append((A[2*j]+A[2*j+1])/2)
            c.append((A[2*j]-A[2*j+1])/2)
        A=a[:]
        C=C+c[::-1]
    return A+C

#Recomposition en ondelettes de Haar

def reHaar1D(L):
    n=len(L)
    A=[L[0]]
    B=L[1:]
    for i in range(valuation(n)):
        C=[]
        for j in range(len(A)):
            m=len(B)
            C.append(A[j]+B[m-1-j])
            C.append(A[j]-B[m-1-j])
        del B[m-1-j:m]
        A=C[:]
    return A

def Final(L) :
    return reHaar1D(Suppression1D(Haar1D(L),epsilon))

p=10
epsilon=1.5

X=np.linspace(0,1,2**p)
Y=f(X)
Z=Final(Y)

X1=np.linspace(0,1,9)
Y1=[X1[0]]
Z1=[g(X1[0])]
for x in X1[1:] :
    Y1.append(x)
    Y1.append(x)
    Z1.append(Z1[-1])
    Z1.append(g(x))

plt.plot(Y1,Z1)
plt.show()

##plt.plot(X,Z)
##plt.show()

U=g(X)
V=Final(U)

plt.plot(X,V)
plt.show()
