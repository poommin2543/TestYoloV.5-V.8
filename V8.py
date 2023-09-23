from ultralytics import YOLO
import cv2

model = YOLO("Path//bestV8.pt")
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    res = model.predict(source = frame, show = True)
    
    # for ans in res:
    #     print(ans.boxes.conf)
    
    cv2.waitKey(1)
cv2.destroyAllWindows()