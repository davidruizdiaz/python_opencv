import cv2

#--- Carga de modelos ---#
# Arquitectura del modelo
arqModel = "model/MobileNetSSD_deploy.prototxt.txt"
# Pesos
model = "model\MobileNetSSD_deploy.caffemodel"
# Etiquetas de la red
classNames = { 0: 'fondo',
    1: 'avión', 2: 'bicicleta', 3: 'pájaro', 4: 'bote',
    5: 'botella', 6: 'autobus', 7: 'automovil', 8: 'gato', 9: 'silla',
    10: 'vaca', 11: 'mesa', 12: 'perro', 13: 'caballo',
    14: 'motocicleta', 15: 'Persona', 16: 'planta',
    17: 'oveja', 18: 'sofá', 19: 'tren', 20: 'televisor' }

# Carga del modelo
net = cv2.dnn.readNetFromCaffe(arqModel, model)

#--- Captura de video ---#
captura = cv2.VideoCapture(0)

# recorre y muestra las capturas de la camara
while (captura.isOpened()) :
    success, imagen, = captura.read()
    if success == True :
        # guarda el alto y el ancho original
        height, width, _ = imagen.shape
        # redimensiona, para usar con el modelo (ver https://github.com/opencv/opencv/blob/b0eddeba2d838163142d5fd1b9cd22ec1bc2a1ac/samples/dnn/models.yml#L52)
        image_resized = cv2.resize(imagen, (300,300))
        # crea un binario de imagen, blob. Recibe 4 params: la imagen, el factor de escala, el tamaño de la imagen y el valor mean para realizar la substracción de la imagen
        blob = cv2.dnn.blobFromImage(image_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5))

        #--- Detección y predicción ---#
        # carga el blob al modelo
        net.setInput(blob)
        # propaga la red y obtiene la detección
        detections = net.forward()
        # recorrer las detecciones
        for detection in detections[0][0] :
            # Para la detección si el objeto es una persona
            if detection[2] > 0.5 :
                # recupera la etiqueta del objeto detectado que se encuenta en la posición 1 de la detección
                label = classNames[detection[1]]
                # crea el rectangulo extrayendo lo últimos 4 elementos de la detección y multiplicandolos
                # por el alto y el ancho original
                box = detection[3:7] * [width, height, width, height]
                # se crean los puntos con valores enteros
                xStart, yStart, xEnd, yEnd = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                # dibuja un rectagulo en la imagen utilizando los puntos
                cv2.rectangle(imagen, (xStart, yStart), (xEnd, yEnd), (0, 255, 0), 2)
                # muestra labels
                cv2.putText(imagen, "Conf: {:.0f}%".format(detection[2] * 100), (xStart, yStart - 5), 1, 1.2, (255, 0, 0), 2)
                cv2.putText(imagen, label, (xStart, yStart - 25), 1, 1.2, (255, 0, 0), 2)
                break
        # muestra las imagenes
        cv2.imshow('Capturas de video', imagen)
        # 0xFF cuando la maquina es de 64bits y verifica
        # que se haya presionado la tecla s
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break
    else:
         break


captura.release()
cv2.destroyAllWindows()


