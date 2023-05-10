from ultralytics import YOLO as yo
import numpy

# caraga de modelo 
model = yo("model/yolov8n.pt", "v8")

# predicción
salida = model.predict(source="img/imagen.jpg", conf=0.28, save=True)

# salida como array de tensor
print(salida)

# salida como array de numpy
print(">>> Salida tipo:",type(salida[0]))
