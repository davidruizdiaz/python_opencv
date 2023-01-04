import cv2 as cv

# captura las imagenes de la camara de la note
captura = cv.VideoCapture(0)

# para grabar el video
# parametros: nombre, codec, fps, tamaño
salida = cv.VideoWriter('mivideo.avi', cv.VideoWriter_fourcc(*'XVID'), 20.0, (640,480))

# recorre y muestra las capturas de la camara
while (captura.isOpened()):
    ret, imagen = captura.read()
    if ret==True:
        cv.imshow('Capturas de video', imagen)

        # para guardar cada imagen en el video
        salida.write(imagen)

        # 0xFF cuando la maquina es de 64bits y verifica
        # que se haya presionado la tecla s
        if cv.waitKey(40) & 0xFF == ord('s'):
            cv.imwrite('ultima.png', imagen)
            break
    else:
         break

# cierra las capturas
captura.release()
salida.release()
cv.destroyAllWindows()
    
