from ultralytics import YOLO

model = YOLO("model/best.pt")

CLASS_MAP = {
    0: "Merchant",   # Merchant
    1: "date",   # date
    2: "total",   # total
    3: "number",   # no
    4: "item"      # item
}

def detect_and_crop(image):

    results = model.predict(source=image, conf=0.35)

    crops = []

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        for box, cls in zip(boxes, classes):
            x1, y1, x2, y2 = map(int, box)

            crop = image[y1:y2, x1:x2]

            crops.append({
                "image": crop,
                "type": CLASS_MAP[int(cls)]
            })

    return crops