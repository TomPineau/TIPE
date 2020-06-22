import matplotlib.image as img

def nivgris(image) :
    for i in range(512) :
        for j in range(512) :
            image[i][j]=(image[i][j][0]+image[i][j][1]+image[i][j][2])/3
    return image

def f(x) :
    return x**2

def ligne(L) :
    C=[]
    for i in range(len(L)[0]) :
        for j in range(len(L)[1]) :
            C.append(L[i][j])
    return C

B=[1,2,3,
   4,5,6,
   7,8,9]

A=img.imread('Lena.png')
