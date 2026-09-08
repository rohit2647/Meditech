import streamlit as st
import torch
import cv2
import numpy as np

from PIL import Image
from io import BytesIO
from datetime import datetime

from ultralytics import YOLO

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
    PageBreak
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
# BRAIN AREA
# ============================================================

def calculate_brain_area(image_rgb):

    gray = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2GRAY
    )

    _, threshold = cv2.threshold(
        gray,
        10,
        255,
        cv2.THRESH_BINARY
    )

    return cv2.countNonZero(threshold)


# ============================================================
# SEVERITY
# ============================================================

def calculate_severity(tumor_percentage):

    if tumor_percentage < 1:

        return "Low", "Routine"

    elif tumor_percentage < 3:

        return "Moderate", "Attention"

    elif tumor_percentage < 5:

        return "High", "High"

    else:

        return "Very High", "Urgent"


# ============================================================
# GRAD-CAM
# ============================================================

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_hook = (
            target_layer.register_forward_hook(
                self.forward_hook_function
            )
        )

        self.backward_hook = (
            target_layer.register_full_backward_hook(
                self.backward_hook_function
            )
        )

    def forward_hook_function(
        self,
        module,
        input_data,
        output
    ):

        self.activations = output

    def backward_hook_function(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = grad_output[0]

    def remove_hooks(self):

        self.forward_hook.remove()
        self.backward_hook.remove()


# ============================================================
# GENERATE GRAD-CAM
# ============================================================

def generate_gradcam(image_rgb):

    try:

        device = next(
            model.model.parameters()
        ).device

        # Same target layer used in your YOLO model
        target_layer = model.model.model[-2]

        gradcam = GradCAM(
            model.model,
            target_layer
        )

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

        model.model.zero_grad()

        output = model.model(
            tensor
        )

        if isinstance(
            output,
            (tuple, list)
        ):

            output_tensor = output[0]

        else:

            output_tensor = output

        if output_tensor.ndim != 3:

            gradcam.remove_hooks()

            return None

        # Handle [B, C, N]
        if output_tensor.shape[1] >= 6:

            positive_scores = (
                output_tensor[:, 5, :]
            )

        # Handle [B, N, C]
        elif output_tensor.shape[2] >= 6:

            positive_scores = (
                output_tensor[:, :, 5]
            )

        else:

            gradcam.remove_hooks()

            return None

        score = positive_scores.max()

        score.backward()

        activations = gradcam.activations

        gradients = gradcam.gradients

        if (
            activations is None
            or gradients is None
        ):

            gradcam.remove_hooks()

            return None

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = (
            weights * activations
        ).sum(
            dim=1
        )

        cam = torch.relu(cam)

        cam = (
            cam
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )

        if cam.max() > 0:

            cam = cam / cam.max()

        cam = cv2.resize(
            cam,
            (
                image_rgb.shape[1],
                image_rgb.shape[0]
            )
        )

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

        overlay = cv2.addWeighted(
            image_rgb,
            0.55,
            heatmap,
            0.45,
            0
        )

        gradcam.remove_hooks()

        return overlay

    except Exception as e:

        st.warning(
            f"Grad-CAM could not be generated: {e}"
        )

        return None


# ============================================================
# PDF REPORT
# ============================================================

def create_pdf(
    detection_image,
    gradcam_image,
    tumor_status,
    confidence,
    grade,
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
        ["Grade", grade],
        ["Dimensions", dimensions_text],
        ["Tumor Area", tumor_area_text],
        ["Brain Area", brain_area_text],
        ["Tumor / Brain Area", percentage_text],
        ["Severity", severity],
        ["Priority", priority]
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

    if gradcam_image is not None:

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
    the detected bounding box is used to estimate tumor
    dimensions and pixel area.

    Brain area is estimated using image thresholding.
    The tumor-to-brain pixel-area ratio is then used for
    an experimental severity and priority categorization.

    Grad-CAM is used to visualize image regions contributing
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

    The current model does not provide tumor grade labels.
    Therefore, Grade is reported as Not Available.

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

            grade = "Not Available"

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

                tumor_area = (
                    width * height
                )

                brain_area = (
                    calculate_brain_area(
                        image_rgb
                    )
                )

                if brain_area > 0:

                    tumor_percentage = (
                        tumor_area /
                        brain_area
                    ) * 100

                    severity, priority = (
                        calculate_severity(
                            tumor_percentage
                        )
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

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Tumor Status",
                "Detected"
                if tumor_detected
                else "Not Detected"
            )

        with col2:

            st.metric(
                "Confidence",
                (
                    f"{best_confidence * 100:.2f}%"
                    if best_confidence is not None
                    else "N/A"
                )
            )

        with col3:

            st.metric(
                "Grade",
                grade
            )

        with col4:

            st.metric(
                "Priority",
                priority
            )

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
                    f"{tumor_percentage:.2f}%"
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
                    caption="Model Attention",
                    width=500
                )

            else:

                st.info(
                    "Grad-CAM visualization "
                    "could not be generated."
                )

        # ----------------------------------------------------
        # GRADE
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">🧬 Tumor Grade</div>',
            unsafe_allow_html=True
        )

        st.info(
            "Grade prediction is currently unavailable. "
            "The YOLO model used here is trained for "
            "positive/negative tumor detection, not "
            "Grade I–IV classification."
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
            grade=grade,
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