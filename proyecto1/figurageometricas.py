import cv2

# carga imagen
image = cv2.imread('figurasColores.png')

# cambia a tono de grices
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detecta bordes
canny = cv2.Canny(gray, 10, 150)
canny = cv2.dilate(canny, None, iterations=1)
canny = cv2.erode(canny,None,iterations=1)

# recupera y dibuja contornos
cnts,_ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(image, cnts, -1, (0,255,0), 2)


for c in cnts:
    epsilon = 0.01*cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c,epsilon,True)
    print("cantidad de vértices = %d" % len(approx))
    x,y,w,h = cv2.boundingRect(approx)

    if len(approx)==3:
        cv2.putText(image, 'Triangulo', (x+10,y-15),1,1,(0,255,0),1)    

    if len(approx)==4:
        aspect_ratio = float(w)/h
        print("-- aspect ratio = %d" % aspect_ratio)
        if aspect_ratio==1:
            cv2.putText(image, 'Cuadrado', (x+10,y-15),1,1,(0,255,0),1)    
        else:
            cv2.putText(image, 'Rectangulo', (x+10,y-15),1,1,(0,255,0),1)    

    if len(approx)==5:
        cv2.putText(image, 'Pentagono', (x+10,y-15),1,1,(0,255,0),1)    

    if len(approx)==6:
        cv2.putText(image, 'Hexanogo', (x+10,y-15),1,1,(0,255,0),1)    

    if len(approx)>=10:
        cv2.putText(image, 'Circulo', (x+10,y-15),1,1,(0,255,0),1)    

    cv2.drawContours(image, [approx], 0, (0,255,0), 2)
    cv2.imshow('image', image)
    cv2.waitKey(0)


cv2.imshow('image', image)
# cv2.imshow('image', gray)
cv2.imshow('image', canny)
cv2.waitKey(0)
cv2.destroyAllWindows()

