import os
import streamlit as st
import torch
import cv2
import numpy as np

from PIL import Image
from io import BytesIO
from datetime import datetime

from ultralytics import YOLO

from src.assessment import (
    estimate_brain_mask,
    calculate_mask_area,
    calculate_tumor_percentage,
    experimental_severity_priority,
)
from src.segmentation import segment_tumor, overlay_mask
from src.classification import load_classifier, classify_tumor

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as PDFImage,
    PageBreak,
    CondPageBreak
)
from reportlab.lib.units import inch


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Meditech",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .hero {
        text-align: center;
        padding: 25px 10px 20px 10px;
    }

    .hero-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .hero-subtitle {
        font-size: 18px;
        color: #888;
        margin-bottom: 10px;
    }

    .upload-box {
        border: 1px dashed #777;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        margin-top: 15px;
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    .info-card {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        min-height: 110px;
    }

    .status-card {
        border-radius: 14px;
        padding: 18px;
        text-align: center;
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 20px;
    }

    .disclaimer {
        border: 1px solid rgba(255,193,7,0.4);
        border-radius: 12px;
        padding: 15px;
        margin-top: 25px;
        font-size: 14px;
    }

    .footer-text {
        text-align: center;
        color: #888;
        font-size: 13px;
        margin-top: 35px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🧠 Meditech</div>
        <div class="hero-subtitle">
            AI-Assisted Brain MRI Tumor Analysis
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_path = "models/best.pt"

    return YOLO(model_path)


try:

    model = load_model()

except Exception as e:

    st.error(
        f"Unable to load the model.\n\n"
        f"Make sure your file exists at:\n"
        f"`models/best.pt`\n\n"
        f"Error: {e}"
    )

    st.stop()


# ============================================================
# OPTIONAL MODELS (SAM segmentation, tumor classifier)
# ============================================================

# Brain area, tumor area and severity now live in src/assessment.py;
# tumor masks come from src/segmentation.py.

SAM_WEIGHT_CANDIDATES = ["models/mobile_sam.pt", "models/sam_b.pt"]


@st.cache_resource
def load_sam():
    """Load SAM only if its weights are present (no silent downloads).
    Without it, segmentation falls back to the bounding box."""

    for path in SAM_WEIGHT_CANDIDATES:

        if os.path.exists(path):

            try:

                from ultralytics import SAM

                return SAM(path)

            except Exception:

                return None

    return None


@st.cache_resource
def load_tumor_classifier():
    """Returns None if models/tumor_classifier.pt does not exist."""

    return load_classifier()


sam_model = load_sam()

tumor_classifier = load_tumor_classifier()


# ============================================================
# GRAD-CAM
# ============================================================

# --- Grad-CAM overlay look (change these numbers to taste) ---
# 1.0 = original scan brightness, lower = darker background
GRADCAM_BACKGROUND_BRIGHTNESS = 0.55
# How strongly the hot regions are colored (0 to 1)
GRADCAM_HEAT_STRENGTH = 0.80
# Attention below this fraction of the maximum is not colored at all
GRADCAM_CUTOFF = 0.10


class GradCAM:
    """Grad-CAM-style attention (HiResCAM variant) over several feature maps.

    YOLO's Detect head reads three feature maps (strides 8, 16, 32).
    The top-scoring detection comes from only ONE of them, and the other
    two receive zero gradient. So we hook all three and add the maps up.
    """

    def __init__(self, target_layers):

        self.target_layers = target_layers

        self.activations = [None] * len(target_layers)

        self.gradients = [None] * len(target_layers)

        self.handles = []

        for index, layer in enumerate(target_layers):

            self.handles.append(
                layer.register_forward_hook(
                    self._make_forward_hook(index)
                )
            )

    def _make_forward_hook(self, index):

        def hook(module, input_data, output):

            self.activations[index] = output

            if output.requires_grad:

                output.register_hook(
                    lambda grad, i=index: self._save_gradient(i, grad)
                )

        return hook

    def _save_gradient(self, index, grad):

        self.gradients[index] = grad

    def remove_hooks(self):

        for handle in self.handles:

            handle.remove()


# ============================================================
# GENERATE GRAD-CAM
# ============================================================

@torch.enable_grad()
def generate_gradcam(image_rgb):

    gradcam = None

    try:

        network = model.model

        device = next(
            network.parameters()
        ).device

        # The feature maps the Detect head actually reads from
        # (layers 16, 19, 22 for YOLO11n).
        detect_head = network.model[-1]

        layer_ids = getattr(detect_head, "f", None)

        if isinstance(layer_ids, (list, tuple)):

            target_layers = [
                network.model[i] for i in layer_ids
            ]

        else:

            target_layers = [network.model[-2]]

        gradcam = GradCAM(target_layers)

        resized = cv2.resize(
            image_rgb,
            (640, 640)
        )

        tensor = torch.from_numpy(
            resized
        ).permute(
            2,
            0,
            1
        ).float() / 255.0

        tensor = tensor.unsqueeze(0)

        tensor = tensor.to(device)

        # YOLO weights are frozen after predict(); the input must
        # require grad or backward() has nothing to flow through.
        tensor.requires_grad_(True)

        network.eval()

        network.zero_grad()

        output = network(tensor)

        if isinstance(output, (tuple, list)):

            output_tensor = output[0]

        else:

            output_tensor = output

        if output_tensor.ndim != 3:

            return None

        # Handle [B, C, N]
        if output_tensor.shape[1] >= 6:

            positive_scores = output_tensor[:, 5, :]

        # Handle [B, N, C]
        elif output_tensor.shape[2] >= 6:

            positive_scores = output_tensor[:, :, 5]

        else:

            return None

        # Score of the single best "positive" detection
        score = positive_scores.max()

        score.backward()

        height, width = image_rgb.shape[:2]

        combined = np.zeros((height, width), dtype=np.float32)

        for activations, gradients in zip(
            gradcam.activations,
            gradcam.gradients
        ):

            # Scales that did not produce the top detection get no
            # gradient and simply contribute nothing.
            if activations is None or gradients is None:

                continue

            # HiResCAM: keep the gradients spatial (element-wise product).
            # Averaging them over the whole map (classic Grad-CAM) spreads
            # a single detection's evidence over every similar-looking
            # region, which is misleading for object detectors.
            cam = torch.relu(
                (gradients * activations).sum(dim=1)
            )

            cam = (
                cam
                .squeeze()
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            combined += cv2.resize(
                cam,
                (width, height)
            )

        if combined.max() <= 0:

            st.warning(
                "Grad-CAM produced no signal for this image."
            )

            return None

        cam = combined / combined.max()

        heatmap = np.uint8(
            255 * cam
        )

        heatmap = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )

        heatmap = cv2.cvtColor(
            heatmap,
            cv2.COLOR_BGR2RGB
        )

        # Blend per pixel: a darkened scan with color only on the hot regions
        # (no blue tint over everything).
        alpha = np.clip(
            (cam - GRADCAM_CUTOFF) / (1 - GRADCAM_CUTOFF), 0, 1
        )[..., None] * GRADCAM_HEAT_STRENGTH

        base = image_rgb.astype(np.float32) * GRADCAM_BACKGROUND_BRIGHTNESS

        overlay = (
            base * (1 - alpha)
            + heatmap.astype(np.float32) * alpha
        ).clip(0, 255).astype(np.uint8)

        return overlay

    except Exception as e:

        st.warning(
            f"Grad-CAM could not be generated: {e}"
        )

        return None

    finally:

        if gradcam is not None:

            gradcam.remove_hooks()


# ============================================================
# PDF REPORT
# ============================================================

def create_pdf(
    detection_image,
    segmentation_image,
    gradcam_image,
    tumor_status,
    confidence,
    classification,
    segmentation_method,
    dimensions,
    tumor_area,
    brain_area,
    tumor_percentage,
    severity,
    priority
):

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "Title",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=25,
        spaceAfter=8
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=11,
        textColor=colors.grey,
        spaceAfter=15
    )

    heading_style = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontSize=16,
        spaceBefore=18,
        spaceAfter=10
    )

    story = []

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "MEDITECH",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Brain MRI Tumor Analysis Report",
            subtitle_style
        )
    )

    story.append(
        Paragraph(
            datetime.now().strftime(
                "Generated: %d %B %Y, %I:%M %p"
            ),
            subtitle_style
        )
    )

    # --------------------------------------------------------
    # RESULT TABLE
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Analysis Results",
            heading_style
        )
    )

    confidence_text = (
        f"{confidence:.2f}%"
        if confidence is not None
        else "N/A"
    )

    dimensions_text = (
        f"{dimensions[0]:.2f} × "
        f"{dimensions[1]:.2f} pixels"
        if dimensions is not None
        else "N/A"
    )

    tumor_area_text = (
        f"{tumor_area:.2f} pixels²"
        if tumor_area is not None
        else "N/A"
    )

    brain_area_text = (
        f"{brain_area:.2f} pixels²"
        if brain_area is not None
        else "N/A"
    )

    percentage_text = (
        f"{tumor_percentage:.2f}%"
        if tumor_percentage is not None
        else "N/A"
    )

    table_data = [
        ["Parameter", "Result"],
        ["Tumor Status", tumor_status],
        ["Detection Confidence", confidence_text],
        ["Tumor Classification", classification],
        ["Dimensions", dimensions_text],
        ["Tumor Area", tumor_area_text],
        ["Segmentation Method", segmentation_method],
        ["Brain Area", brain_area_text],
        ["Tumor / Brain Area", percentage_text],
        ["Severity", severity],
        ["Priority", priority]
    ]

    if classification == "Not Available":

        table_data = [
            row for row in table_data
            if row[0] != "Tumor Classification"
        ]

    table = Table(
        table_data,
        colWidths=[
            2.5 * inch,
            3.5 * inch
        ]
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#1F4E79")
                ),

                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white
                ),

                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold"
                ),

                (
                    "FONTNAME",
                    (0, 1),
                    (0, -1),
                    "Helvetica-Bold"
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),

                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    8
                )
            ]
        )
    )

    story.append(table)

    # --------------------------------------------------------
    # DETECTION IMAGE
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Tumor Detection",
            heading_style
        )
    )

    detection_buffer = BytesIO()

    Image.fromarray(
        detection_image
    ).save(
        detection_buffer,
        format="PNG"
    )

    detection_buffer.seek(0)

    story.append(
        PDFImage(
            detection_buffer,
            width=5.8 * inch,
            height=4.2 * inch
        )
    )

    # --------------------------------------------------------
    # GRAD-CAM
    # --------------------------------------------------------

    if segmentation_image is not None:

        story.append(CondPageBreak(5.8 * inch))

        story.append(
            Paragraph(
                "Tumor Segmentation",
                heading_style
            )
        )

        segmentation_buffer = BytesIO()

        Image.fromarray(
            segmentation_image
        ).save(
            segmentation_buffer,
            format="PNG"
        )

        segmentation_buffer.seek(0)

        seg_height, seg_width = segmentation_image.shape[:2]

        story.append(
            PDFImage(
                segmentation_buffer,
                width=5.0 * inch,
                height=5.0 * inch * seg_height / seg_width
            )
        )

    if gradcam_image is not None:

        story.append(CondPageBreak(5.8 * inch))

        story.append(
            Paragraph(
                "Grad-CAM Explainability",
                heading_style
            )
        )

        gradcam_buffer = BytesIO()

        Image.fromarray(
            gradcam_image
        ).save(
            gradcam_buffer,
            format="PNG"
        )

        gradcam_buffer.seek(0)

        story.append(
            PDFImage(
                gradcam_buffer,
                width=5.8 * inch,
                height=4.2 * inch
            )
        )

    # --------------------------------------------------------
    # METHODOLOGY
    # --------------------------------------------------------

    story.append(PageBreak())

    story.append(
        Paragraph(
            "Methodology",
            heading_style
        )
    )

    methodology = """
    The MRI image is analyzed using a YOLO-based brain tumor
    detection model. When a positive detection is obtained,
    the detected bounding box is used to estimate tumor dimensions and to prompt a segmentation step (SAM when available, otherwise the bounding box) that produces the tumor mask used for the pixel area.

    Brain area is estimated with Otsu thresholding, largest-component selection and hole filling.
    The tumor-to-brain pixel-area ratio is then used for
    an experimental severity and priority categorization.

    A Grad-CAM variant (HiResCAM, gradient x activation) is used to visualize image regions contributing
    to the model's prediction.
    """

    story.append(
        Paragraph(
            methodology,
            styles["BodyText"]
        )
    )

    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Important Disclaimer",
            heading_style
        )
    )

    disclaimer = """
    Meditech is an AI-assisted research prototype and is not
    a clinically validated diagnostic device.

    The severity and priority classifications are experimental
    research outputs and should not be interpreted as medical
    diagnosis or treatment recommendations.

    Tumor masks are algorithmic estimates, not expert-drawn segmentations. Tumor classification is reported only when a separately trained classifier is installed.

    Physical measurements in millimeters cannot be determined
    reliably without MRI pixel-spacing information.
    """

    story.append(
        Paragraph(
            disclaimer,
            styles["BodyText"]
        )
    )

    document.build(story)

    buffer.seek(0)

    return buffer


# ============================================================
# UPLOAD SECTION
# ============================================================

st.markdown(
    '<div class="section-title">📤 Upload MRI Scan</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="upload-box">
        <b>Upload a brain MRI image</b><br>
        <span style="color:#888">
        Supported formats: JPG, JPEG, PNG
        </span>
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose MRI image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ],
    label_visibility="collapsed"
)


# ============================================================
# PREVIEW
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    image_rgb = np.array(image)

    st.markdown(
        '<div class="section-title">🖼️ MRI Preview</div>',
        unsafe_allow_html=True
    )

    # CENTERED IMAGE
    col1, col2, col3 = st.columns(
        [1, 2, 1]
    )

    with col2:

        st.image(
            image_rgb,
            caption="Uploaded MRI Scan",
            width=500
        )

    st.write("")

    # ========================================================
    # ANALYZE BUTTON
    # ========================================================

    analyze = st.button(
        "🔍 Analyze MRI",
        type="primary",
        use_container_width=True
    )

    if analyze:

        with st.spinner(
            "Analyzing MRI... Please wait."
        ):

            # ------------------------------------------------
            # YOLO
            # ------------------------------------------------

            results = model.predict(
                image_rgb,
                conf=0.25,
                verbose=False
            )

            result = results[0]

            tumor_detected = False

            best_confidence = None

            positive_box = None

            # ------------------------------------------------
            # FIND POSITIVE DETECTION
            # ------------------------------------------------

            if result.boxes is not None:

                for box in result.boxes:

                    class_id = int(
                        box.cls[0].item()
                    )

                    confidence_value = float(
                        box.conf[0].item()
                    )

                    # Your dataset:
                    #
                    # 0 = negative
                    # 1 = positive

                    if class_id == 1:

                        if (
                            best_confidence is None
                            or
                            confidence_value >
                            best_confidence
                        ):

                            best_confidence = (
                                confidence_value
                            )

                            positive_box = box

                            tumor_detected = True

            # ------------------------------------------------
            # DEFAULT VALUES
            # ------------------------------------------------

            classification = "Not Available"

            classification_ranked = None

            segmentation_image = None

            segmentation_method = "N/A"

            box_area = None

            dimensions = None

            tumor_area = None

            brain_area = None

            tumor_percentage = None

            severity = "No tumor detected"

            priority = "Routine"

            # ------------------------------------------------
            # TUMOR DETECTED
            # ------------------------------------------------

            if tumor_detected:

                x1, y1, x2, y2 = (
                    positive_box
                    .xyxy[0]
                    .cpu()
                    .numpy()
                )

                width = x2 - x1

                height = y2 - y1

                dimensions = (
                    width,
                    height
                )

                tumor_box = (x1, y1, x2, y2)

                box_area = width * height

                # Brain mask: Otsu + largest component + hole filling
                brain_mask = estimate_brain_mask(image_rgb)

                brain_area = calculate_mask_area(brain_mask)

                # Tumor mask: SAM if weights exist, otherwise the bounding box itself
                tumor_mask, segmentation_method = segment_tumor(
                    image_rgb,
                    tumor_box,
                    brain_mask,
                    sam_model
                )

                tumor_area = calculate_mask_area(tumor_mask)

                segmentation_image = overlay_mask(
                    image_rgb,
                    tumor_mask
                )

                # Optional tumor type / grade classifier
                classified = classify_tumor(
                    tumor_classifier,
                    image_rgb,
                    tumor_box
                )

                if classified is not None:

                    classification = (
                        f"{classified[0]} ({classified[1]:.1f}%)"
                    )

                    classification_ranked = classified[2]

                if brain_area > 0:

                    tumor_percentage = calculate_tumor_percentage(
                        tumor_area,
                        brain_area
                    )

                    severity, priority = experimental_severity_priority(
                        tumor_percentage
                    )

            # ------------------------------------------------
            # DETECTION IMAGE
            # ------------------------------------------------

            detection_image = result.plot()

            detection_image = cv2.cvtColor(
                detection_image,
                cv2.COLOR_BGR2RGB
            )

            # ------------------------------------------------
            # GRAD-CAM
            # ------------------------------------------------

            gradcam_image = generate_gradcam(
                image_rgb
            )

        # ====================================================
        # RESULTS
        # ====================================================

        st.markdown(
            '<div class="section-title">📊 Analysis Results</div>',
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        if tumor_detected:

            st.markdown(
                f"""
                <div class="status-card"
                     style="
                     background:#3b1114;
                     border:1px solid #8b3036;
                     ">
                    ⚠️ Tumor Detected
                    <br>
                    <small>
                    Confidence:
                    {best_confidence * 100:.2f}%
                    </small>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="status-card"
                     style="
                     background:#10351f;
                     border:1px solid #28734a;
                     ">
                    ✓ No Tumor Detected
                </div>
                """,
                unsafe_allow_html=True
            )

        # ----------------------------------------------------
        # RESULT CARDS
        # ----------------------------------------------------

        cards = [
            (
                "Tumor Status",
                "Detected" if tumor_detected else "Not Detected"
            ),
            (
                "Confidence",
                (
                    f"{best_confidence * 100:.2f}%"
                    if best_confidence is not None
                    else "N/A"
                )
            ),
        ]

        # Show Classification only when a classifier produced a result
        if classification_ranked is not None:

            cards.append(("Classification", classification))

        cards.append(("Priority", priority))

        for card_column, (card_label, card_value) in zip(
            st.columns(len(cards)),
            cards
        ):

            with card_column:

                st.metric(card_label, card_value)

        # ----------------------------------------------------
        # MEASUREMENTS
        # ----------------------------------------------------

        if tumor_detected:

            st.markdown(
                '<div class="section-title">📐 Tumor Measurements</div>',
                unsafe_allow_html=True
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "Width",
                    f"{dimensions[0]:.2f} px"
                )

            with col2:

                st.metric(
                    "Height",
                    f"{dimensions[1]:.2f} px"
                )

            with col3:

                st.metric(
                    "Tumor Area",
                    f"{tumor_area:.2f} px²"
                )

            with col4:

                st.metric(
                    "Tumor / Brain",
                    (
                        f"{tumor_percentage:.2f}%"
                        if tumor_percentage is not None
                        else "N/A"
                    )
                )

            # ------------------------------------------------
            # SEVERITY
            # ------------------------------------------------

            st.markdown(
                '<div class="section-title">⚠️ Severity Assessment</div>',
                unsafe_allow_html=True
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Severity",
                    severity
                )

            with col2:

                st.metric(
                    "Priority",
                    priority
                )

            st.warning(
                "Severity and priority are experimental "
                "research outputs based on tumor-to-brain "
                "pixel-area ratio. They are not clinically "
                "validated."
            )

        # ----------------------------------------------------
        # VISUALIZATIONS
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">🔬 AI Visualization</div>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                "### 🎯 Tumor Detection"
            )

            st.image(
                detection_image,
                caption="YOLO Detection",
                width=500
            )

        with col2:

            st.markdown(
                "### 🔥 Grad-CAM"
            )

            if gradcam_image is not None:

                st.image(
                    gradcam_image,
                    caption="Model Attention (gradient × activation)",
                    width=500
                )

            else:

                st.info(
                    "Grad-CAM visualization "
                    "could not be generated."
                )

        # ----------------------------------------------------
        # SEGMENTATION
        # ----------------------------------------------------

        if tumor_detected:

            st.markdown(
                '<div class="section-title">🧩 Tumor Segmentation</div>',
                unsafe_allow_html=True
            )

            col1, col2 = st.columns(2)

            with col1:

                st.image(
                    segmentation_image,
                    caption=f"Tumor mask ({segmentation_method})",
                    width=500
                )

            with col2:

                st.metric(
                    "Mask area",
                    f"{tumor_area:.0f} px²"
                )

                st.metric(
                    "Bounding-box area",
                    f"{box_area:.0f} px²"
                )

            if segmentation_method.startswith("Bounding box"):

                st.warning(
                    "The tumor could not be separated from its "
                    "surroundings, so the bounding box was used. "
                    "Area and tumor/brain percentage are likely "
                    "overestimates."
                )

        # ----------------------------------------------------
        # CLASSIFICATION
        # ----------------------------------------------------

        if tumor_detected and classification_ranked is not None:

            st.markdown(
                '<div class="section-title">🧬 Tumor Classification</div>',
                unsafe_allow_html=True
            )

            if classification_ranked is not None:

                st.success(
                    f"Predicted class: {classification}"
                )

                for label, class_confidence in classification_ranked:

                    st.write(
                        f"{label}: {class_confidence:.1f}%"
                    )

                    st.progress(
                        min(class_confidence / 100, 1.0)
                    )

                st.warning(
                    "Experimental output from a separately trained "
                    "classifier. Not clinically validated."
                )

        # ----------------------------------------------------
        # PDF REPORT
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">📄 Report</div>',
            unsafe_allow_html=True
        )

        pdf = create_pdf(
            detection_image=detection_image,
            segmentation_image=segmentation_image,
            gradcam_image=gradcam_image,
            tumor_status=(
                "Tumor Detected"
                if tumor_detected
                else "No Tumor Detected"
            ),
            confidence=(
                best_confidence * 100
                if best_confidence is not None
                else None
            ),
            classification=classification,
            segmentation_method=segmentation_method,
            dimensions=dimensions,
            tumor_area=tumor_area,
            brain_area=brain_area,
            tumor_percentage=tumor_percentage,
            severity=severity,
            priority=priority
        )

        st.download_button(
            label="📄 Download Complete PDF Report",
            data=pdf,
            file_name=(
                "Meditech_Brain_Tumor_Report.pdf"
            ),
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )

        # ----------------------------------------------------
        # DISCLAIMER
        # ----------------------------------------------------

        st.markdown(
            """
            <div class="disclaimer">
            <b>⚠️ Research Prototype Disclaimer</b><br><br>
            Meditech is an AI-assisted research prototype
            and is not a clinically validated diagnostic
            system. Results should not be used as a substitute
            for professional medical evaluation.
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer-text">
        Meditech • AI-Assisted Brain MRI Analysis<br>
        Research Prototype
    </div>
    """,
    unsafe_allow_html=True
)