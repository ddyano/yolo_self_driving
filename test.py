import cv2
from ultralytics import YOLO
import time

model = YOLO(r"C:\Users\hp\runs\detect\train5\weights\best.pt")

url = "http://192.168.1.3:8080/video"

cap = cv2.VideoCapture(url)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

frame_count = 0
last_command = ""
last_shape_time = 0
shape_cooldown = 3
last_arrow_time = 0
arrow_cooldown = 2
while True:
    ret, frame = cap.read()

    if not ret:
        continue

    frame_count += 1

    # skip frames
    if frame_count % 3 != 0:
        continue

    frame = cv2.resize(frame, (480, 360))
    frame_center = frame.shape[1] / 2

    results = model(frame, imgsz=416, device=0, verbose=False)

    # read detections
    boxes = results[0].boxes
    largest_border = None
    largest_height = 0
    arrow_detected = False
    for box in boxes:
    
        cls = int(box.cls[0])
        label = model.names[cls]
    
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    
        height = y2 - y1
        center_x = (x1 + x2) / 2
        
        width = x2 - x1
        area = width * height
    
        current_time = time.time()

        if label == "circle" and area > 20000:
            if current_time - last_shape_time > shape_cooldown:
                print("Circle detected nearby")
                last_shape_time = current_time
        
        if label == "rectangle" and area > 20000:
            if current_time - last_shape_time > shape_cooldown:
                print("Rectangle detected nearby")
                last_shape_time = current_time
        
        if label == "border":

            if height > largest_height:
                largest_height = height
                largest_border = center_x

        if label == "arrow" and area > 12000:

            if current_time - last_arrow_time > arrow_cooldown:
        
                arrow_detected = True
        
                if center_x < frame_center:
                    print("TURN LEFT")
                else:
                    print("TURN RIGHT")
        
                last_arrow_time = current_time
                
    # make steering decision once per frame
    if not arrow_detected and largest_border is not None and largest_height > 200:
    
        print("Border too close")
    
        command = "RIGHT" if largest_border < frame_center else "LEFT"
    
        if command != last_command:
            print("STEER", command)
            last_command = command   
    annotated = results[0].plot()
    
    cv2.imshow("YOLO Detection", annotated)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()