# Meditech

Brain tumor detection, segmentation and explainability project.

## Structure

- `app.py` - Streamlit app: detection, tumor segmentation, optional classification, Grad-CAM, PDF report.
- `notebooks/yolo_model.ipynb` - YOLO training / transfer learning (positive / negative detector).
- `notebooks/grad_cam.ipynb` - YOLO inference and Grad-CAM.
- `models/best.pt` - trained YOLO detector weights.
- `models/mobile_sam.pt` - *optional* SAM weights for better tumor masks (see below).
- `models/tumor_classifier.pt` - *optional* tumor type / grade classifier (see below).
- `src/assessment.py` - brain mask, mask areas, tumor/brain percentage, experimental severity/priority.
- `src/segmentation.py` - tumor mask inside the detected box (SAM, else GrabCut, else box).
- `src/classification.py` - optional classifier wrapper.
- `train_classifier.py` - trains the optional classifier.
- `data/` - local datasets only; do not commit datasets unless licensing permits.
- `outputs/` - predictions and Grad-CAM results.

## How the analysis works

1. **Detection** - YOLO finds `positive` / `negative` boxes.
2. **Brain mask** - Otsu threshold, largest connected component, hole filling.
3. **Tumor mask** - the detection box is refined into a pixel mask:
   SAM if `models/mobile_sam.pt` (or `models/sam_b.pt`) exists, otherwise GrabCut.
   If neither can separate the tumor, the box is used and the app says so.
   Tumor area and tumor/brain percentage are computed from this mask.
4. **Classification (optional)** - if `models/tumor_classifier.pt` exists, the padded tumor
   crop is classified; otherwise the result is reported as Not Available.
5. **Grad-CAM** - shows which regions drove the detector's prediction.

### Enabling SAM masks
Download MobileSAM weights (e.g. via Ultralytics: `from ultralytics import SAM; SAM("mobile_sam.pt")`),
then copy the file to `models/mobile_sam.pt`.

### Enabling classification
Prepare `data/tumor_classification/{train,val}/<class_name>/*.jpg`, run
`python train_classifier.py`. Class folders can be tumor types or grades - the app shows
whatever the folder names are. You need a dataset that actually has those labels, split by patient.

## Important

- Severity and priority thresholds are experimental project rules, not clinically validated criteria.
- GrabCut and SAM masks are algorithmic estimates, not expert segmentations. They were only
  tested on synthetic images, not on real MRI. For trustworthy areas, train a segmentation model
  (e.g. YOLO11-seg or nnU-Net) on expert-drawn masks.
- The detector is weak (positive-class precision 0.31, mAP50 0.38 on validation); detections
  should be treated as low-confidence.
- The classifier is only as good as its training data; no model is included.
- This is a research prototype, not a diagnostic device.
