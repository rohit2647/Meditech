"""Train the tumor-type / grade classifier used by src/classification.py.

Expected folder layout (one sub-folder per class, one split per folder):

    data/tumor_classification/
        train/
            glioma/        *.jpg
            meningioma/    *.jpg
            pituitary/     *.jpg
        val/
            glioma/ ...
            meningioma/ ...
            pituitary/ ...

Use class folders such as grade_1 ... grade_4 instead if your dataset
has grade labels. You need a dataset that actually contains those
labels (e.g. a public brain tumor type dataset, or BraTS-derived grade
labels). The detector's dataset only has positive/negative.

Split by PATIENT, not by image, or validation accuracy will be
inflated by near-duplicate slices from the same scan.

Run:  python train_classifier.py
Then copy the best weights to models/tumor_classifier.pt
"""

import shutil
from pathlib import Path

from ultralytics import YOLO

DATA_DIR = "data/tumor_classification"
EPOCHS = 30
IMAGE_SIZE = 224


def main():
    model = YOLO("yolo11n-cls.pt")
    model.train(
        data=DATA_DIR,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        project="runs_classifier",
        name="tumor_classifier",
        exist_ok=True,
    )

    best = Path("runs_classifier/tumor_classifier/weights/best.pt")
    if best.exists():
        Path("models").mkdir(exist_ok=True)
        shutil.copy(best, "models/tumor_classifier.pt")
        print("Saved models/tumor_classifier.pt")

    metrics = model.val()
    print("Top-1 accuracy:", metrics.top1)


if __name__ == "__main__":
    main()
