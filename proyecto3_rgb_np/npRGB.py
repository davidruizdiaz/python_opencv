import cv2 as cv
import numpy as np


# se crea una matriz tridimensional de 300x300
# para crear la imagen
# bgr=np.zeros((300,300,3), dtype=np.uint8)

bgr = cv.imread('loro.png')

# modifica los colores según rgb pero con las posiciones bgr
# bgr[:,:,:] = (255,0,0)

# captura cada componente de la matriz 
b=bgr[:,:,0]
g=bgr[:,:,1]
r=bgr[:,:,2]

# muestra la imagen
# cv.imshow('BGR', bgr)
cv.imshow('BGR', np.hstack([b,g,r]))

# convierte a rgb
rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
# captura cada componente de la matriz 
r=rgb[:,:,0]
g=rgb[:,:,1]
b=rgb[:,:,2]

# muestra la imagen
cv.imshow('RGB', np.hstack([r,g,b]))

gris = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
cv.imshow('GRIS', gris)

cv.waitKey(0)
cv.destroyAllWindows()
