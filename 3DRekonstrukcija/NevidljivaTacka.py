import cv2
import easygui
import unicodedata

def homogeneUAfine(koord):
    return (koord[0] / koord[2], koord[1] / koord[2], 1)

def osmoteme(vektor):
    def vektorskiProizvod(v1, v2):
        x1 = v1[1]*v2[2] - v1[2]*v2[1]
        x2 = v1[2]*v2[0] - v1[0]*v2[2]
        x3 = v1[0]*v2[1] - v1[1]*v2[0]

        return (x1, x2, x3)
    v1, v2, v3, v5, v6, v7, v8 = vektor

    v21 = vektorskiProizvod(v2, v1)
    v65 = vektorskiProizvod(v6, v5)
    v78 = vektorskiProizvod(v7, v8)
    
    Yb1 = vektorskiProizvod(v21, v65)
    Yb2 = vektorskiProizvod(v21, v78)
    Yb3 = vektorskiProizvod(v65, v78)

    Yb1 = homogeneUAfine(Yb1)
    Yb2 = homogeneUAfine(Yb2)
    Yb3 = homogeneUAfine(Yb3)

    Ybx1 = (Yb1[0] + Yb2[0] + Yb3[0])/3
    Ybx2 = (Yb1[1] + Yb2[1] + Yb3[1])/3

    Yb = (Ybx1, Ybx2, 1)

    v73 = vektorskiProizvod(v7, v3)
    v62 = vektorskiProizvod(v6, v2)
    v51 = vektorskiProizvod(v5, v1)

    Xb1 = vektorskiProizvod(v73, v62)
    Xb2 = vektorskiProizvod(v73, v51)
    Xb3 = vektorskiProizvod(v62, v51)

    Xb1 = homogeneUAfine(Xb1)
    Xb2 = homogeneUAfine(Xb2)
    Xb3 = homogeneUAfine(Xb3)

    Xbx1 = (Xb1[0] + Xb2[0] + Xb3[0])/3
    Xbx2 = (Xb1[1] + Xb2[1] + Xb3[1])/3

    Xb = (Xbx1, Xbx2, 1)

    vYb3 = vektorskiProizvod(Yb, v3)
    vXb8 = vektorskiProizvod(Xb, v8)


    tacka = vektorskiProizvod(vYb3, vXb8)
    return [round(tacka[0]/tacka[2]), round(tacka[1]/tacka[2])]

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
 
        font = cv2.FONT_HERSHEY_SIMPLEX

        tacke.append((x, y, 1))
        if(len(tacke) % 7 == 0):
            trazena = osmoteme(tacke)
            cv2.putText(img, '.' + str(trazena[0]) + ',' + str(trazena[1]), (trazena[0],trazena[1]), font, 0.5, (0, 0, 255), 2)
            tacke.clear()
        cv2.putText(img, '.' + str(x) + ',' + str(y), (x,y), font, 0.5, (255, 0, 0), 2)
        cv2.imshow('image', img)
 
if __name__=="__main__":
 
    code = easygui.fileopenbox()
    path = unicodedata.normalize('NFKD', code).encode('ascii','ignore')
    img = cv2.imread(code, 1)
    img = cv2.resize(img, (1200, 1600))
    cv2.imshow('image', img)
 
    tacke = []

    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()