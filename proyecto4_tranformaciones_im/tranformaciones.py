import cv2 as cv
import numpy as np
import imutils

imagen = cv.imread('./ave.jpg')

ancho = imagen.shape[1] # columnas
alto = imagen.shape[0] # filas

# Tranlación
MTra = np.float32([[1,0,10],[0,1,100]])
# parametros: imagen, matriz de tranlación, tamaño
imagenTra = cv.warpAffine(imagen,MTra,(ancho,alto))

# Rotación cv2.getRotationMatrix2D(...)
# parámetros: 
#     centro de rotación, 
#     ángulo de rotación en grados (rotación en sentido antihorario),
#     escala, valor de escala isotropica
MRot = cv.getRotationMatrix2D((ancho//2, alto//2), 15, 1)
imagenRot = cv.warpAffine(imagen,MRot,(ancho,alto))

# Escalado cv2.resize(...)
# parámetros:
#     imagen de entrada
#     (ancho, alto) que se quiere dar a la nueva imagen
#     metodo de interpolación
imagenEs = cv.resize(imagen, (ancho+150, alto+150), interpolation=cv.INTER_CUBIC)
# escalado con imutils
imagenEsIm = imutils.resize(imagen, width=400)

# Recorte, se debe probar con la matriz de la imagen
print('Imagen shape: ', imagen.shape) # para saber las medidas de la matriz

# utilizar 
# x1,y1 vertice superior izquierdo
# x2,y2 vertice inferior derecho
# seleccionar el rango a cortar de la imagen[y1:y2, x1:x2]
# (x1=280,y1=50) (x2=470,y2=200)
imagenRec = imagen[50:200, 280:470] 


cv.imshow('Entrada', imagen)
cv.imshow('Salida - translacion', imagenTra)
cv.imshow('Salida - rotacion', imagenRot)
cv.imshow('Salida - escalada', imagenEs)
cv.imshow('Salida - escalada - imutils', imagenEsIm)
cv.imshow('Salida - recorte', imagenRec)

cv.waitKey(0)
cv.destroyAllWindows()
