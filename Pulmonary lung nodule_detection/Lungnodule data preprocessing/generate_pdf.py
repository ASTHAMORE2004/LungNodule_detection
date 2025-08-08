import os
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
from tensorflow.keras.models import load_model
from data_loader import loading_the_data, change_label_names, get_generators
from utils.grad_cam_utils import make_gradcam_heatmap, overlay_heatmap
from config import DATA_DIR

# ─────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────
model = load_model("lung_cancer_model.h5")

df = loading_the_data(DATA_DIR)
change_label_names(df)
_, _, test_gen = get_generators(df)
class_names = list(test_gen.class_indices.keys())

# Find last conv layer (EfficientNetB3 specific)
from tensorflow.keras.layers import Conv2D
last_conv_layer_name = None
for layer in reversed(model.layers):
    if isinstance(layer, Conv2D):
        last_conv_layer_name = layer.name
        break
assert last_conv_layer_name is not None, "No conv layer found!"

# ─────────────────────────────────────────────────────────
# PDF Setup
# ─────────────────────────────────────────────────────────
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.set_font("Arial", size=12)

# ─────────────────────────────────────────────────────────
# Generate Report for 5 Images
# ─────────────────────────────────────────────────────────
print("📄 Generating PDF Report...")

batch_x, _ = next(test_gen)

os.makedirs("temp_images", exist_ok=True)

for i in range(min(5, len(batch_x))):
    img = batch_x[i]
    img_rgb = (img * 255).astype("uint8")

    # Prediction
    preds = model.predict(img[np.newaxis, ...], verbose=0)
    pred_class = np.argmax(preds[0])
    confidence = preds[0][pred_class]

    # Grad-CAM
    heatmap = make_gradcam_heatmap(img[np.newaxis, ...], model, last_conv_layer_name, pred_class)
    cam_img = overlay_heatmap(img_rgb, heatmap)

    # Save cam image
    img_path = f"temp_images/gradcam_{i}.jpg"
    plt.imsave(img_path, cam_img)

    # Add to PDF
    pdf.add_page()
    pdf.cell(200, 10, txt=f"Test Image #{i+1}", ln=True, align='L')
    pdf.image(img_path, w=100)
    pdf.ln(85)
    pdf.cell(200, 10, txt=f"Predicted: {class_names[pred_class]}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence: {confidence:.2f}", ln=True)

# ─────────────────────────────────────────────────────────
# Save PDF
# ─────────────────────────────────────────────────────────
pdf.output("Lung_Cancer_Report.pdf")
print("✅ PDF saved as Lung_Cancer_Report.pdf")
