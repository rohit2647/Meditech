# Meditech

Brain tumor detection and explainability project.

## Structure

- `notebooks/yolo_training.ipynb` - YOLO training / transfer learning.
- `notebooks/gradcam.ipynb` - YOLO inference and Grad-CAM.
- `models/best.pt` - trained YOLO weights. Add your actual `best.pt` here.
- `src/assessment.py` - tumor area, tumor/brain percentage, and experimental severity/priority logic.
- `data/` - local datasets only; do not commit the full dataset unless licensing permits.
- `outputs/` - predictions and Grad-CAM results.
- `requirements.txt` - Python dependencies.

## Important

The current severity and priority thresholds are experimental project rules, not clinically validated medical criteria.

The current tumor-area calculation based on YOLO bounding boxes is an approximation. A segmentation model would be needed for a more accurate tumor-pixel area.
