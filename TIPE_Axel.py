import numpy as np
import matplotlib.image as img
from copy import deepcopy
import matplotlib.pyplot as plt
C=np.array([[2,0,1,1,0,2,0,0],[4,2,2,2,4,0,2,2],[4,4,0,4,2,4,0,2],[4,0,0,0,4,4,3,3],[6,0,4,6,6,2,2,6],[2,4,6,6,6,0,5,5],[2,0,0,0,0,0,2,2],[4,4,6,6,2,2,0,0]])
P=img.imread('Lenna.png')
L= np.reshape(np.arange(262144),(512,512))
epsilon=0.02


def Suppression(M,epsilon):
    """ Cette fonction supprime les coefficients de M inferieurs a epsilon"""
    n=len(M)
    for i in range(n) :
        if abs(M[i])<epsilon:
            M[i]=0
    return M

def nivgris(image):
    """ Cette fonction transforme une image en couleur en Noir et Blanc"""
    (n,p,q)=np.shape(image)
    res=np.zeros((n,p))
    for i in range(n):
        for j in range(p) :
            moy=(image[i,j,0]+image[i,j,1]+image[i,j,2])/3
            res[i,j]=moy
    return res

def Zeros(M):
    """ Cette fonction compte le nombre de zero dans M"""
    i=0
    n=len(M)
    for j in range(n):
        if M[j]==0:
            i+=1
    return i

def etape_suivante(A):
    n,p=np.shape(A)
    return A[:(n//2),:(p//2)]

def recuperation(A):
    n,p=np.shape(A)
    L=[]
    M=[]
    N=[]
    for i in range(n//2):
        for j in range(p//2):
            L.append(A[i][(n//2)+j])
            M.append(A[(n//2)+i][j])
            N.append(A[(n//2)+i][(n//2)+j])
    return L+M+N

def valuation(s):
    i=0
    while s%2==0:
        i+=1
        s=s//2
    return i

def MatricesHaar(n):
    """ Cette fonction cree la matrice de Haar de taille nxn"""
    A=np.zeros((n,n))
    j=0
    for i in range(0,n,2):
        A[i][j]=1/2
        A[i+1][j]=1/2
        A[i][(n//2)+j]=1/2
        A[i+1][(n//2)+j]=-1/2
        j+=1
    return A

def tMatricesHaar(n):
    """ Cette fonction cree la transposee de la matrice de Haar de taille nxn"""
    A=np.zeros((n,n))
    i=0
    for j in range(0,n,2):
        A[i][j]=1/2
        A[i][j+1]=1/2
        A[(n//2)+i][j]=1/2
        A[(n//2)+i][j+1]=-1/2
        i+=1
    return A

def Haar(A):
    """Cette fonction donne la decomposition en ondelettes de Haar de A"""
    x,y=np.shape(A)
    a=valuation(x)
    z=0
    B=deepcopy(A)
    F=[]
    while z<a:
        n,p=np.shape(B)
        C=MatricesHaar(n)
        tC=tMatricesHaar(n)
        D=np.dot(B,C)#effectue un produit matriciel 
        E=np.dot(tC,D)#effectue un produit matriciel 
        B=etape_suivante(E)#cette fonction garde la partie de la matrice
                           #dont on a besoin pour l'etape suivante
                           # elle est de taille (n//2)x(n//2)
        F+=recuperation(E)#cette fonction garde la partie de la matrice
                          #dont on ne se sert pas a l'etape suivante
        z+=1
    return B,Suppression(F,epsilon)#B est la moyenne de tous les coefficients de A
                                   #et F est la decomposition de A sur la base de Haar

def tMatricesHaar2(n):
    """ Cette fonction cree l'inverse de la matrice de Haar de taille nxn"""
    A=np.zeros((n,n))
    i=0
    for j in range(0,n,2):
        A[i][j]=1
        A[i][j+1]=1
        A[(n//2)+i][j]=1
        A[(n//2)+i][j+1]=-1
        i+=1
    return A

def MatricesHaar2(n):
    """ Cette fonction cree la transposee de l'inverse de la matrice de Haar de taille nxn"""
    A=np.zeros((n,n))
    j=0
    for i in range(0,n,2):
        A[i][j]=1
        A[i+1][j]=1
        A[i][(n//2)+j]=1
        A[i+1][(n//2)+j]=-1
        j+=1
    return A

def fusion(A,B,C,D):
    """Cette fonction fusionne les 4 matrices pour en faire une seule"""
    res1=np.concatenate((A,B),axis=1)
    res2=np.concatenate((C,D),axis=1)
    return np.concatenate((res1,res2),axis=0)
    
def RecompositionMatrice(B,F):
    """Cette fonction recompose la matrice de depart a partir de sa valeur moyenne
et sa decomposition sur la base de Haar"""
    n=len(B)
    l=len(F)
    a=valuation(int((n+l)**(1/2)))
    k=0
    while k < a:
        n,p=np.shape(B)
        A,C,E=F[-n**2:],F[-2*n**2:-n**2],F[-3*n**2:-2*n**2]
        A,C,E=np.reshape(A,(n,n)),np.reshape(C,(n,n)),np.reshape(E,(n,n))
        F=F[:-3*n**2]
        Z=fusion(B,E,C,A)
        H=np.dot(Z,tMatricesHaar2(2*n))
        B=np.dot(MatricesHaar2(2*n),H)
        k+=1
    return B


def MatriceVBR(M):
    """Cette fonction renvoie 3 matrices a partir de M,
les matrices representent les coefficients de M en vert/bleu/rouge"""
    n,p,q=np.shape(M)
    N,O,P=np.zeros((n,p)),np.zeros((n,p)),np.zeros((n,p))
    for i in range(n):
        for j in range(p):
            N[i,j],O[i,j],P[i,j]=M[i,j,0],M[i,j,1],M[i,j,2]
    return N,O,P

def MatriceCouleur(P,O,N):
    """Cette fonction renvoie une matrice en couleur a partir
des matrices vert/bleu/rouge"""
    n,p=np.shape(O)
    W=np.zeros((n,p,3))
    for i in range(n):
        for j in range(p):
            W[i,j,0]=P[i,j]
            W[i,j,1]=O[i,j]
            W[i,j,2]=N[i,j]
    return W



def HaarCouleur(M):
    N,O,P=MatriceVBR(M)
    N1,N2=Haar(N)
    O1,O2=Haar(O)
    P1,P2=Haar(P)
    N,O,P=RecompositionMatrice(N1,N2),RecompositionMatrice(O1,O2),RecompositionMatrice(P1,P2)
    return MatriceCouleur(N,O,P)




E=img.imread('Etienne.png')
plt.imshow(HaarCouleur(E))
plt.show()
            
Lena=nivgris(P)
plt.imshow(Lena,cmap=plt.cm.gray)
plt.show()

X,Y=Haar(Lena)
S=RecompositionMatrice(X,Y)
plt.imshow(S,cmap=plt.cm.gray)
plt.show()

plt.imshow(P)
plt.show()
plt.imshow(HaarCouleur(P))
plt.show()
