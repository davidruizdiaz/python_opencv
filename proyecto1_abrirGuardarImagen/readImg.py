import cv2 as cv

# imread lee la imagen, si se le agrega el parámetro 0
# muestra en escala de grices
imagen = cv.imread('./logo.png')

# imshow muestra la imagen
cv.imshow('Prueba de imagen', imagen)

# imwrite guarda la imagen
cv.imwrite('imagen nueva.png', imagen)

# cierra la imagen al presionar una tecla
cv.waitKey(0)

# cierra las ventanas
cv.destroyAllWindows()

