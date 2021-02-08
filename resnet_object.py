from imageai.Detection import ObjectDetection
import os
import cv2


execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()

cap=cv2.VideoCapture(0)

while(1):
    #detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "image.jpeg"), output_image_path=os.path.join(execution_path , "imagenew.jpg"))
    r,frame=cap.read()
    detections = detector.detectObjectsFromImage(frame)
    for eachObject in detections:
        print(eachObject["name"] , " : " , eachObject["percentage_probability"] )





#pip3 install https://github.com/OlafenwaMoses/ImageAI/releases/download/2.0.2/imageai-2.0.2-py3-none-any.whl

#Install image AI
