# YOLO Model Weights Directory

This folder holds model weights for:
1. `yolov8n.pt` - Ultralytics standard object detection (cars, motorcycles, persons, trucks, buses)
2. `custom_helmet.pt` - Custom trained helmet & no-helmet detection model (optional)
3. `custom_plate.pt` - Custom license plate detection model (optional)

### Automatic Download
When `main.py` runs, `ultralytics` automatically downloads `yolov8n.pt` if not already present in this directory.

You can also manually download weights using:
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```
or place your fine-tuned `best.pt` file here as `custom_helmet.pt`.

