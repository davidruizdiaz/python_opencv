import cv2 as cv
import numpy as np

imagen = cv.imread('./ave.jpg')

ancho = imagen.shape[1] # columnas
alto = imagen.shape[0] # filas

# Tranlación
M = np.float32([[1,0,10],[0,1,100]])
# parametros: imagen, matriz de tranlación, tamaño
imagenOut = cv.warpAffine(imagen,M,(ancho,alto))

cv.imshow('Entrado', imagen)
cv.imshow('Salida', imagenOut)
cv.waitKey(0)
cv.destroyAllWindows()
