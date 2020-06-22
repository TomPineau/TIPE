import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
from copy import deepcopy

A=np.array([[1,2],[3,4]])
B=np.array([[1,2,3,4],[2,3,1,4],[2,8,5,7],[1,9,8,3]])
C=np.array([[2,0,1,1,0,2,0,0],[4,2,2,2,4,0,2,2],[4,4,0,4,2,4,0,2],[4,0,0,0,4,4,3,3],[6,0,4,6,6,2,2,6],[2,4,6,6,6,0,5,5],[2,0,0,0,0,0,2,2],[4,4,6,6,2,2,0,0]])
D=np.array([[1,2,3,4],[2,3,3,1],[1,1,2,1],[3,1,2,3]])
E=[2.0625,-0.3125,0.3125,-0.0625,-0.5,-0.5,0,0.25,0.75,-0.75,0.5,-0.5,-0.5,0,-0.5,0.5]

epsilon=0.2
p=16

#En 2 dimensions

def sign(x) :
    if x>=0 :
        return 1
    else :
        return -1
    
def nivgris(image) :
    n=len(image)
    M=np.zeros((n,n))
    for i in range(n) :
        for j in range(n) :
            M[i][j]=(image[i][j][0]+image[i][j][1]+image[i][j][2])/3
    return M

def Suppression2D_dur(L,epsilon):
    """L est une liste et epsilon un seuil à partir duquel toutes les
valeurs de L inférieures à epsilon sont réduites à 0"""
    n=len(L)
    for i in range(n) :
        if abs(L[i])<epsilon:
            L[i]=0
    return L

def Suppression2D_doux(L,epsilon):
    n=len(L)
    for i in range(n) :
        if abs(L[i])<epsilon:
            L[i]=0
        else :
            L[i]=L[i]-sign(L[i])*epsilon
    return L

def Zeros(L):
    """L est une liste et on renvoie le nombre de zéros que la
liste L contient"""
    s=0
    n=len(L)
    for i in range(n) :
        if L[i]==0:
            s+=1
    return s

def ZerosHaar2D(M) :
    return Zeros(Suppression2D(Haar2D(M),epsilon))

#Décomposition en ondelettes de Haar 

def Matrice2D(M) :
    """M est un tableau de dimension (2,2) et on renvoie sa décomposition en
ondelettes de Haar"""
    return np.array([[(M[0][0]+M[0][1]+M[1][0]+M[1][1])/4,(M[0][0]-M[0][1]+M[1][0]-M[1][1])/4],[(M[0][0]+M[0][1]-M[1][0]-M[1][1])/4,(M[0][0]-M[0][1]-M[1][0]+M[1][1])/4]])
    
def Haar2Dbis(M) :
    """M est un tableau de dimension (2**n,2**n). On renvoie une liste A
constituée de tous les coefficients en haut à gauche de chacun des
tableaux de dimension (2,2) décomposés en ondelettes de Haar qui
composent M, et une liste C constituée de tous les autres coefficients
présents dans chacun des tableaux de dimension (2,2) décomposés en
ondelettes de Haar qui composent M"""
    (n,p)=np.shape(M)
    A=[]
    C=[]
    for i in range(0,n,2) :
        for j in range(0,p,2) :
            L=Matrice2D(np.array([[M[i][j],M[i][j+1]],[M[i+1][j],M[i+1][j+1]]]))
            A.append(L[0][0])
            C.append(L[0][1])
            C.append(L[1][0])
            C.append(L[1][1])
    return A,C
    
def Haar2D(M) :
    """M est un tableau de dimension (2**n,2**n) et tant que la liste
A n'est pas reduite à un singleton, on appelle la fonction Haar2Dbis
appliquée à A. On somme toutes les autres listes C"""
    (n,p)=np.shape(M)    
    A=Haar2Dbis(M)[0]
    C=Haar2Dbis(M)[1]
    while len(A)!=1 :
        l=len(A)
        B=np.reshape(A,(int(l**(1/2)),int(l**(1/2))))
        A=Haar2Dbis(B)[0]
        C=Haar2Dbis(B)[1]+C
        l=int(l**(1/2))
    return A+C

#Recomposition en ondelettes de Haar
    
def reMatrice2D(L) :
    """L est une liste de longueur 4 et on renvoie la recomposition par
le principe des ondelettes de Haar"""
    return [L[0]+L[1]+L[2]+L[3],L[0]-L[1]+L[2]-L[3],L[0]+L[1]-L[2]-L[3],L[0]-L[1]-L[2]+L[3]]
    
def reHaar2Dbis(L) :
    """L est une liste de longueur 2**n, A est une liste de longueur
(2**n)//4 et C une liste de longueur 3*(2**n)//4. On parcourt chacun des
coefficients de la liste A où chaque coefficient de A correspond trois
coefficients de la liste C, on reunit ces quatre coefficients dans une
liste sur laquelle on applique la recomposition par le principe des
ondelettes de Haar. On réorganise ensuite cette liste en passant par un
tableau de dimension (2**(n/2),2**(n/2))"""
    l=len(L)
    A,C=L[:l//4],L[l//4:]
    Z=[0]*l
    i,j,k=0,0,0
    while i!=l//4 and j!=(3*l)//4 and k!=l :
        M=reMatrice2D([A[i],C[j],C[j+1],C[j+2]])
        Z[k:k+4]=[M[0],M[1],M[2],M[3]]
        i+=1
        j+=3
        k+=4
    m=int(l**(1/2))
    n=0
    M=np.zeros((m,m))
    for a in range(0,m,2) :
        for b in range(0,m,2) :
            M[a:a+2,b:b+2]=np.reshape(Z[n:n+4],(2,2))
            n+=4
    F=[]
    for u in range(m) :
        for v in range(m) :
            F.append(M[u][v])
    return F
    
def reHaar2D(L) :
    """L est une liste de longueur 2**n, A est une liste de longueur
4 correspondant à L[:4] sur laquelle on applique la recomposition par
le principe des ondelettes de Haar où à chaque boucle la taille de A
est multipliée par 4 et on la recompose jusqu'à ce que A soit de la
même longueur que L"""
    K=L[:]
    l=len(K)
    s=4
    A=reHaar2Dbis(K[:4])
    Z=[0]*l
    m=int(l**(1/2))
    while s//4!=l :
        Z[:s]=A
        A=reHaar2Dbis(A+K[s:4*s])
        s=4*s
    return np.reshape(Z,(m,m))
    
def Final(M) :
    """M est un tableau sur lequel on applique la decomposition en
ondelettes de Haar,la suppression de coefficients négligeables
et la recomposition par le principe des ondelettes de Haar"""
    return reHaar2D(Suppression2D_dur(Haar2D(M),epsilon))

def Final_doux(M) :
    return reHaar2D(Suppression2D_doux(Haar2D(M),epsilon))

#En couleur

def Separation(M) :
    (n,p,q)=np.shape(M)
    R,V,B=np.zeros((n,p)),np.zeros((n,p)),np.zeros((n,p))
    for i in range(n):
        for j in range(p):
            R[i,j],V[i,j],B[i,j]=M[i,j,0],M[i,j,1],M[i,j,2]
    return R,V,B

def Restitution(R,V,B) :
    n,p=np.shape(V)
    M=np.zeros((n,p,3))
    for i in range(n):
        for j in range(p):
            M[i,j,0]=R[i,j]
            M[i,j,1]=V[i,j]
            M[i,j,2]=B[i,j]
    return M

def reHaar2D_couleur(M) :
    R,V,B=Separation(M)
    return Restitution(Final(R),Final(V),Final(B))

def affine(M) :
    (n,p,q)=np.shape(M)
    for i in range(n) :
        for j in range(p) :
            for k in range(q) :
                if M[i][j][k]>1 :
                    M[i][j][k]=1
                elif M[i][j][k]<0 :
                    M[i][j][k]=0
    return M

def affiche(n) :
    if n==1 :
        plt.imshow(affine(reHaar2D_couleur(P)))
        plt.show()
        print("hey")
    elif n==2 :
        plt.imshow(Final(R),cmap=plt.cm.gray)
        plt.show()
    elif n==3 :
        plt.imshow(Final_doux(R),cmap=plt.cm.gray)
        plt.show()

def test(M,eps) :
    D=Haar2D(M)
    print(eps,Zeros(Suppression2D(D,eps)),len(D))
    plt.imshow(reHaar2D(Suppression2D(D,eps)),cmap=plt.cm.gray)
    plt.show()


#Z1=Suppression2D_dur(np.linspace(-1,1,2**p),epsilon)
#Z2=Suppression2D_doux(np.linspace(-1,1,2**p),epsilon)
#plt.plot(np.linspace(-1,1,2**p),Z1)
#plt.show()
#plt.plot(np.linspace(-1,1,2**p),Z2)
#plt.show()
    
P=img.imread('Lenna.png')       #1
R=nivgris(P)                    #2
affiche(1)
affiche(2)
affiche(3)
