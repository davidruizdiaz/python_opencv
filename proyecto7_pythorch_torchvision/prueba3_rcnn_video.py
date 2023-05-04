import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision import transforms as T
import cv2

pesos = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
categorias = pesos.meta["categories"]
# print(categorias)
model = fasterrcnn_resnet50_fpn(weights = pesos, box_score_thresh=0.45)
# configura el modo evaluacion
model.eval()
# print(model)

cap = cv2.VideoCapture("img/traffic.mp4")

while True :
    success, frame = cap.read()
    if success == True :
        height, width, _ = frame.shape
        imgResize = cv2.resize(frame, (300, 300))
        imgResize = cv2.cvtColor(imgResize, cv2.COLOR_BGR2RGB)
        imgTensor = T.functional.to_tensor(imgResize)

        with torch.no_grad():
            pred = model([imgTensor])

        boxes, labels, scores = pred[0]["boxes"], pred[0]["labels"], pred[0]["scores"]
        c=0
        for i in boxes:
            xTop, yTop, xDown, yDown = i.numpy().astype('int') 
            etiqueta = categorias[labels.numpy()[c]]
            c+=1
            imgResize = cv2.rectangle(imgResize, (xTop, yTop), (xDown, yDown), (0, 255, 0), 1)
            imgResize = cv2.putText(imgResize, etiqueta, (xTop, yTop-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1, cv2.LINE_AA)

        cv2.imshow("Video", imgResize)
        if cv2.waitKey(1) & 0xFF == ord('s') :
            break
    else :
        break


cv2.destroyAllWindows()
