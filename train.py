from ultralytics import YOLO

def main():

    # load pretrained YOLOv8 nano model
    model = YOLO("yolov8n.pt")

    # start training
    model.train(
        data="data.yaml",   # dataset config
        epochs=50,          # training epochs
        imgsz=640,          # image size
        batch=8,            # safe for RTX 3050 4GB
        device=0,           # use GPU
        workers=2           # stable for Windows
    )

if __name__ == "__main__":
    main()