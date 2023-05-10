import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision import transforms as T

from PIL import Image
import cv2

pesos = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
categorias = pesos.meta["categories"]
# print(categorias)
model = fasterrcnn_resnet50_fpn(weights = pesos, box_score_thresh=0.45)
# configura el modo evaluacion
model.eval()
# print(model)
pilImg = Image.open("img/vehicles.jpg")
# print(pilImg)
transfoToTensor = T.ToTensor()
tensorImg = transfoToTensor(pilImg)
# print(tensorImg)
with torch.no_grad():
    pred = model([tensorImg])

# print(pred)
# print(pred[0]["boxes"])
# print(pred[0]["labels"])
boxes, labels, scores = pred[0]["boxes"], pred[0]["labels"], pred[0]["scores"]
# etiquetasDetectadas = [categorias[i] for i in labels]
# print(etiquetasDetectadas)
cvImage = cv2.imread("img/vehicles.jpg")

# print(labels)
# print(boxes)
# print(labels.numpy())
c=0;
for i in boxes:
    # print(i)
    xTop, yTop, xDown, yDown = i.numpy().astype('int')
    # print(xTop, yTop, xDown, yDown)
    etiqueta = categorias[labels.numpy()[c]]
    # print(etiqueta)
    c+=1
    cvImage = cv2.rectangle(cvImage, (xTop, yTop), (xDown, yDown), (0, 255, 0), 4)
    cvImage = cv2.putText(cvImage, etiqueta, (xTop, yTop-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2, cv2.LINE_AA)


cvImage = cv2.resize(cvImage, (1000, 583))

while True:
    cv2.imshow("Resultado", cvImage)
    if cv2.waitKey(1) & 0xFF == ord('s') :
        break

# cv2.destroyAllWindows()
