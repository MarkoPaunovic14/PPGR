import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

#c - caj
Lc0 = [693, 373, 1]
Lc1 = [224, 723, 1]
Lc2 = [12, 580, 1]
Lc3 = [500, 288, 1]
Lc4 = [677, 600, 1]
Lc5 = [297, 944, 1]
Lc6 = [124, 816, 1]
Lc7 = [512, 522, 1]

#k - kocka
Lk0 = [1174, 464, 1]
Lk1 = [1017, 565, 1]
Lk2 = [815, 435, 1]
Lk3 = [971, 351, 1]
Lk4 = [1084, 652, 1]
Lk5 = [951, 746, 1]
Lk6 = [778, 621, 1]
Lk7 = [919, 537, 1]


Rc0 = [694, 309, 1]
Rc1 = [160, 563, 1]
Rc2 = [26, 428, 1]
Rc3 = [536, 223, 1]
Rc4 = [683, 539, 1]
Rc5 = [247, 796, 1]
Rc6 = [1225, 661, 1]
Rc7 = [552, 444, 1]

Rk0 = [1171, 437, 1]
Rk1 = [969, 517, 1]
Rk2 = [807, 379, 1]
Rk3 = [996, 316, 1]
Rk4 = [1087, 633, 1]
Rk5 = [915, 707, 1]
Rk6 = [777, 570, 1]
Rk7 = [945, 505, 1]

leve8 = [Lc0, Lc1, Lc2, Lc3, Lk0, Lk1, Lk2, Lk3]
desne8 = [Rc0, Rc1, Rc2, Rc3, Rk0, Rk1, Rk2, Rk3]
Lc = [Lc0, Lc1, Lc2, Lc3, Lc4, Lc5, Lc6, Lc7]
Lk = [Lk0, Lk1, Lk2, Lk3, Lk4, Lk5, Lk6, Lk7]
Rc = [Rc0, Rc1, Rc2, Rc3, Rc4, Rc5, Rc6, Rc7]
Rk = [Rk0, Rk1, Rk2, Rk3, Rk4, Rk5, Rk6, Rk7]

# transformacija piksela
for i in range(8):
    leve8[i][0] = 1200 - leve8[i][0]
    desne8[i][0] = 1200 - desne8[i][0]

LL = np.array(leve8)
RR = np.array(desne8)

def jednacina(l, d):
    x1 = l[0]
    y1 = l[1]
    z1 = l[2]

    x2 = d[0]
    y2 = d[1]
    z2 = d[2]

    return np.array([x1*x2, y1*x2, z1*x2, 
                     x1*y2, y1*y2, z1*y2, 
                     x1*z2, y1*z2, z1*z2])

def koso2v(A):
    return [A[2, 1], A[0, 2], A[1, 0]]

def jednacina2(l, d, T1, T2):
    return np.array([l[1]*T1[2] - l[2]*T1[1],
                    -l[0]*T1[2] + l[2]*T1[0],
                    d[1]*T2[2] - d[2]*T2[1],
                    -d[0]*T2[2] + d[2]*T2[0]])

def TriD(l, d, T1, T2):
    U, S, V = np.linalg.svd(jednacina2(l, d, T1, T2))
    P = V[-1]
    P = P / P[3]
    return P[:-1]

#  prikaz scene - koristi "plotly" biblioteku
def prikazKocke(temenaKocke, ivice): 
    # izdvajamo x,y,z koordinate svih tacaka
    xdata = (np.transpose(temenaKocke))[0]
    ydata = (np.transpose(temenaKocke))[1]
    zdata = (np.transpose(temenaKocke))[2]
    # u data1 ubacujemo sve sto treba naccrtati
    data1 = []
    # za svaku ivicu crtamo duz na osnovu koordinata
    for i in range(len(ivice)):
        data1.append(go.Scatter3d(x=[xdata[ivice[i][0]], xdata[ivice[i][1]]], y=[ydata[ivice[i][0]], ydata[ivice[i][1]]],z=[zdata[ivice[i][0]], zdata[ivice[i][1]]]))
    fig = go.Figure(data = data1 )
    # da ne prikazuje legendu
    fig. update_layout(showlegend=False)
    fig.show()
    # pravi html fajl (ako zelite da napravite "rotatable" 3D rekonstruciju)
    # birate kao parametar velicinu apleta. fulhtml=False je vazno da ne bi pravio ogroman fajl
    # ovde stavite neki vas folder
    fig.write_html("/home/markopaunovic/Desktop/Faks/ppgr/poslednjiDomaci/3D", include_plotlyjs = 'cdn', default_width = '1200px', default_height = '1500px', full_html = False) #Modifiy the html file
    fig.show()

def main():
    jednacine = []
    for i in range(8):  
        jednacine.append(jednacina(LL[i], RR[i]))

    U, S, V = np.linalg.svd(jednacine)

    FF = np.zeros((3, 3))
    V = V[:][-1]

    for i in range(3):
        for j in range(3):
            FF[i][j] = V[3*i+j]

    print("Fundamentalna matrica:")
    print(FF)

    U, S, V = np.linalg.svd(FF)

    e1 = V[:][-1] 
    e1 = (1/e1[2]) * e1 

    e2 = U.T[:][-1]
    e2 = (1/e2[2]) * e2 

    # osnovna matrica EE
    K1 = np.array([[1272.84999, -5.77089, 1020.41693], [0, 1269.10616, 834.36905], [0, 0, 1]])
    EE = K1.T.dot(FF).dot(K1)

    print("Osnovna matrica:")
    print(EE)

    #dekompozicija EE
    Q0 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    E0 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])

    U, SS, V = np.linalg.svd(EE)

    EC = U.dot(E0).dot(U.T)
    AA = U.dot(Q0.T).dot(V.T)

    print("Matrica E0:")
    print(E0)
    print("Matrica A")
    print(AA)

    #triangulacija
    T2 = np.hstack((K1, np.array([[0], [0], [0]])))

    CC = koso2v(EC)
    CC1 = -np.dot(AA.T, CC)

    tmp = np.dot(K1, AA.T)
    dodatak = np.vstack((tmp.T, np.dot(K1, CC1)))

    T1 = dodatak.T

    print("Matrica kamere T1:")
    print(T1)
    print("Matrica kamere T2:")
    print(T2)

    #koordinate u prostoru
    rekonstrukcijaC = []
    rekonstrukcijaK = []

    for i in range(len(Lc)):
        rekonstrukcijaC.append(TriD(Lc[i], Rc[i], T1, T2))

    for i in range(len(Lk)):
        rekonstrukcijaK.append(TriD(Lk[i], Rk[i], T1, T2))
    
    print("Koordinate u prostoru:")
    print(rekonstrukcijaC)
    print(rekonstrukcijaK)

    iviceC = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
    iviceK = [[8, 9], [9, 10], [10, 11], [11, 8], [12, 13], [13, 14], [14, 15], [15, 12], [8, 12], [9, 13], [10, 14], [11, 15]]


    prikazKocke(rekonstrukcijaC, iviceC)
    prikazKocke(rekonstrukcijaK, iviceK)
    

if __name__ == "__main__":
    main()
