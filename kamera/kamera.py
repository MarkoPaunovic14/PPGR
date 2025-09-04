import numpy as np
from numpy import linalg
from scipy import linalg
np.set_printoptions(precision=5, suppress=True)


pts2D =  np.array([1,-1,-1])*(np.array([1600,0,0]) -  np.array([[596, 587, 1], [852, 729, 1], [591, 915, 1], [330, 723, 1], [373, 965,1], [585, 1158,1], [810, 976,1], [592, 780, 1]]))
pts3D = np.array([[0, 0, 3, 1], [0, 3, 3, 1], [3, 3, 3, 1], [3, 0, 3, 1], [3, 0, 0, 1], [3, 3, 0, 1], [0, 3, 0, 1], [2, 2, 3, 1]])

def centar(T):
    R = T[:3, :3]
    t = T[:3, 3]

    C = -np.dot(linalg.inv(R), t)
    C = np.append(C, 1)

    C = np.where(np.isclose(C, 0) , 0.0 , C)
    return C

def kameraK(T):
    R = T[:3, :3]
    
    K, R_new = linalg.rq(R)
    
    K = K / K[2, 2]
    
    for i in range(3):
        if K[i, i] < 0:
            K[:, i] *= -1
 
    K = np.where(np.isclose(K, 0) , 0.0 , K)
    return K

def kameraA(T):
    t0 = np.delete(T, 3, 1)

    if(np.linalg.det(t0) < 0):
        t0 = np.delete(-T, 3, 1)
        
    t0i = np.linalg.inv(t0)
    Q, R = np.linalg.qr(t0i)

    if(R[0, 0] < 0):
        R = np.matmul(np.diag([-1, 1, 1]), R)
        Q = np.matmul(Q, np.diag([-1, 1, 1]))
    if(R[1, 1] < 0):
        R = np.matmul(np.diag([1, -1, 1]), R)
        Q = np.matmul(Q, np.diag([1, -1, 1]))
    if(R[2, 2] < 0):
        R = np.matmul(np.diag([1, 1, -1]), R)
        Q = np.matmul(Q, np.diag([1, 1, -1]))

    A = Q

    A = np.where(np.isclose(A, 0) , 0.0 , A)
    return A

def dveJednacine(img, orig):
    nula = np.array([0,0,0,0])
    prva = np.array(np.concatenate((nula, -img[2]*orig, img[1]*orig)))
    druga = np.array(np.concatenate((img[2]*orig, nula, -img[0]*orig)))
    return prva, druga

def napraviMatricu(imgs, origs):
    num_points = imgs.shape[0]
    
    A = np.zeros((2*num_points, 12))
    
    for i in range(num_points):
        prva, druga = dveJednacine(imgs[i], origs[i])
        A[2*i] = prva
        A[2*i + 1] = druga

    return A

def matricaKamere(imgs, origs):
    a = napraviMatricu(imgs, origs)
    a = np.array(a)
    U, S, Vh = np.linalg.svd(a)
    t = Vh[11]
    t = (1/t[11])*t
    t = t.reshape(3, 4)

    t = t / t[2, 3]
    
    t = np.where(np.isclose(t, 0), 0.0 , t)
    return t

T = matricaKamere(pts2D, pts3D)
print("Matrica kamere:")
print(T)
print()

print("Pozicija centra kamere:")
print(centar(T))
print()

print("Matrica kalibracije kamere:")
print(kameraK(T))
print()

print("Spoljasnja matrica kamere:")
print(kameraA(T))
