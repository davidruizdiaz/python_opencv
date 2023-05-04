import torch
import torchvision
# from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision import transforms as T
import cv2

# pesos = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
pesos = SSD300_VGG16_Weights.DEFAULT
categorias = pesos.meta["categories"]
# print(categorias)
# model = ssdlite320_mobilenet_v3_large(weights=pesos, score_thresh=0.15, nms_thresh=0.15, detections_per_img=40, iou_thresh=0.15)
model = ssd300_vgg16(weights=pesos, score_thresh=0.33, nms_thresh=0.33, detections_per_img=50, iou_thresh=0.33, topk_candidates=50)
# configura el modo evaluacion
model.eval()
# print(model)

cap = cv2.VideoCapture("img/traffic.mp4")

while True :
    success, frame = cap.read()
    if success == True :
        height, width, _ = frame.shape
        imgResize = cv2.resize(frame, (700, 420))
        imgResize = cv2.cvtColor(imgResize, cv2.COLOR_BGR2RGB)
        # imgResize = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgTensor = T.functional.to_tensor(imgResize)

        with torch.no_grad():
            pred = model([imgTensor])
            imgResize = cv2.cvtColor(imgResize, cv2.COLOR_RGB2BGR)

        boxes, labels, scores = pred[0]["boxes"], pred[0]["labels"], pred[0]["scores"]
        c=0
        for i in boxes:
            xTop, yTop, xDown, yDown = i.numpy().astype('int')  
            # xTop = width * xTop
            # yTop = height * yTop
            # xDown = width * xDown
            # yDown = height * yDown
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
