import cv2
import torch
from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from torchvision import transforms as T

# img = read_image("img/perro.jpg")
cap = cv2.VideoCapture('./img/traffic.mp4')

# Paso 1: Inicializa modelo 
pesos = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
modelo = fasterrcnn_resnet50_fpn_v2(weights=pesos, box_score_thresh=0.4)
modelo.eval()

# Paso 2: Inicializa la transformación 
preproceso = pesos.transforms()

while True: 
    success, frame = cap.read()
    if success == True :
        height, width, _ = frame.shape
        image_resized = cv2.resize(frame, (300,300))
        image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        image_tensor = T.functional.to_tensor(image_resized)

        # batch = [preproceso(image_tensor)]
        # Paso 4: Usa el modelo para predecir
        # prediction = modelo(batch)[0]
        with torch.no_grad():
            prediction = modelo([image_tensor])
            print(prediction)
        # labels = [pesos.meta["categories"][i] for i in prediction["labels"]]
        # print(labels)
        # box = draw_bounding_boxes(frame, boxes=prediction["boxes"],
                                  # labels=labels,
                                  # colors="red",
                                  # width=4, font_size=30)
        # im = to_pil_image(box.detach())
        cv2.imshow('Capturas de video', frame)
        # pil_img.show()
        if cv2.waitKey(1) & 0xFF == ord('s') :
            break
    else :
        break

cap.release()
cv2.destroyAllWindows()

