import numpy as np
import math

import plotly.graph_objects as go
import plotly.express as px

# U ovom slucaju crtamo jednostavnu kocku. Vi cete koristiti rekonstruisana temena
temenaKocke =  np.array([[0,0, 3], [0, 3,3], [3,3,3],[3, 0,3],[0,0,0], [0, 3,0], [3, 3,0],[3,0,0]] )
# Ovo su ivice kocke zadate indeksima temena
ivice = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]

# dodatne mogucnosti istrazite sami

#  prikaz scene - koristi "plotly" biblioteku
def prikazKocke(): 
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
    fig.write_html("c:/Users/vasfolder/test.html", include_plotlyjs = 'cdn', default_width = '800px', default_height = '600px', full_html = False) #Modifiy the html file
    fig.show()

prikazKocke()