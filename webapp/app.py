import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import os
import cv2
from PIL import Image
import matplotlib.cm as cm
import plotly.graph_objects as go

st.set_page_config(page_title="Chest X-Ray Classifier", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=DM+Sans:wght@400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .stApp {
        background-color: #0a0e1a;
        color: #e2e8f0;
    }
    .stSidebar {
        background-color: #0f1525 !important;
        border-right: 1px solid #1e2d4a;
    }
    h1, h2, h3 {
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em;
    }
    .metric-card {
        background: linear-gradient(135deg, #0f1b2d 0%, #0f1525 100%);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 8px;
        min-height: 90px;
    }
    .metric-card .label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        color: #4a7fa5;
        text-transform: uppercase;
        letter-spacing: 0.10em;
        margin-bottom: 8px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .metric-card .value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 24px;
        font-weight: 600;
        color: #38bdf8;
        white-space: nowrap;
    }
    .metric-card .value span { color: #38bdf8; }
    .section-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        color: #4a7fa5;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 1px solid #1e2d4a;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid #1e2d4a;
        border-radius: 10px;
        overflow: hidden;
    }
    .stSelectbox > div > div {
        background-color: #0f1525 !important;
        border: 1px solid #1e3a5f !important;
        color: #e2e8f0 !important;
    }
    .stButton > button {
        background: linear-gradient(90deg, #0ea5e9, #2563eb) !important;
        color: white !important;
        border: none !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
    }
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0a0e1a",
    plot_bgcolor="#0f1525",
    font=dict(family="Syne, sans-serif", color="#94a3b8", size=12),
    margin=dict(l=20, r=20, t=40, b=20),
)

@st.cache_resource
def load_keras_model(model_name):
    model_path = os.path.join("models", f"{model_name}.keras")
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

@st.cache_resource
def load_unet_model():
    unet_path = os.path.join("models", "unet_lung_seg.hdf5")
    if not os.path.exists(unet_path):
        unet_path = os.path.join("models", "unet_model.keras")
    if os.path.exists(unet_path):
        return tf.keras.models.load_model(unet_path, compile=False)
    return None

def preprocess_image(image, model_name, target_size=(224, 224), unet_model=None):
    img_array = np.array(image)
    name_lower = model_name.lower()

    if "unet" in name_lower and unet_model is not None:
        img_uint8 = np.clip(img_array, 0, 255).astype(np.uint8)
        unet_input_shape = unet_model.input_shape
        unet_img_size = (unet_input_shape[1], unet_input_shape[2])
        unet_channels = unet_input_shape[3]
        img_unet = cv2.resize(img_uint8, unet_img_size) / 255.0
        if unet_channels == 1:
            img_unet_gray = cv2.cvtColor((img_unet * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            img_unet = np.expand_dims(img_unet_gray / 255.0, axis=-1)
        img_tensor = np.expand_dims(img_unet, axis=0).astype(np.float32)
        mask = unet_model(img_tensor, training=False)[0].numpy()
        mask_resized = cv2.resize(mask, (img_array.shape[1], img_array.shape[0]))
        if len(mask_resized.shape) == 3:
            mask_resized = np.squeeze(mask_resized)
        mask_binary = (mask_resized > 0.5).astype(np.uint8)
        masked_img = cv2.bitwise_and(img_uint8, img_uint8, mask=mask_binary)
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            pad = 10
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(img_array.shape[1], x + w + pad), min(img_array.shape[0], y + h + pad)
            masked_img = masked_img[y1:y2, x1:x2]
        img_array = cv2.resize(masked_img, target_size)

    elif "crop" in name_lower:
        img_uint8 = np.clip(img_array, 0, 255).astype(np.uint8)
        h, w = img_uint8.shape[:2]
        y1, y2 = int(h * 0.15), int(h * 0.90)
        x1, x2 = int(w * 0.05), int(w * 0.95)
        img_array = cv2.resize(img_uint8[y1:y2, x1:x2], target_size)

    else:
        img_array = np.array(image.resize(target_size))

    if "clahe" in name_lower:
        img_uint8 = np.clip(img_array, 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        img_array = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)

    display_image = Image.fromarray(np.uint8(img_array))
    img_array = np.expand_dims(img_array.astype(np.float32), axis=0)

    if "resnet" in name_lower or "densenet" in name_lower:
        img_array = img_array / 255.0
    elif "efficientnet" in name_lower:
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    elif "mobilenet" in name_lower:
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    elif "vgg" in name_lower:
        img_array = tf.keras.applications.vgg16.preprocess_input(img_array)

    return img_array, display_image

def get_last_conv_layer_name(model_name):
    name_lower = model_name.lower()
    if "resnet" in name_lower:         return "conv5_block3_out"
    elif "densenet" in name_lower:     return "relu"
    elif "efficientnet" in name_lower: return "top_activation"
    elif "mobilenet" in name_lower:    return "out_relu"
    elif "vgg" in name_lower:          return "block5_conv3"
    return None

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        inputs = tf.cast(img_array, tf.float32)
        last_conv_layer_output, preds = grad_model(inputs)
        if isinstance(preds, list):
            preds = preds[0]
        class_channel = preds[0][0] if preds.shape[-1] == 1 else preds[:, tf.argmax(preds[0])]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = last_conv_layer_output[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_gradcam(img, heatmap, alpha=0.4):
    img_array = np.array(img)
    jet_heatmap = cm.get_cmap("jet")(np.uint8(255 * heatmap))[:, :, :3]
    jet_heatmap = np.array(
        tf.keras.utils.array_to_img(jet_heatmap).resize((img_array.shape[1], img_array.shape[0]))
    )
    return tf.keras.utils.array_to_img(jet_heatmap * alpha + img_array)

# --- UI fejléc ---
st.markdown("<h1 style='color:#e2e8f0; margin-bottom:4px; font-family:DM Sans,sans-serif; font-weight:700; letter-spacing:-0.01em;'>Chest X-Ray Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#4a7fa5; font-family:JetBrains Mono,monospace; font-size:11px; letter-spacing:0.12em;'>DEEP LEARNING DIAGNOSTIC SYSTEM</p>", unsafe_allow_html=True)

st.sidebar.markdown("<h2 style='color:#e2e8f0;'>Navigation</h2>", unsafe_allow_html=True)
page = st.sidebar.radio("", ["Model Statistics", "Real-time Inference"])

if page == "Model Statistics":

    metrics_dir = "metrics"
    data = []

    if os.path.exists(metrics_dir):
        for filename in sorted(os.listdir(metrics_dir)):
            if filename.endswith(".json"):
                with open(os.path.join(metrics_dir, filename), 'r') as f:
                    try:
                        raw = json.load(f)
                        report = raw.get("classification_report", {})
                        data.append({
                            "Model":            raw.get("model", filename.replace(".json", "")),
                            "Accuracy":         report.get("accuracy", 0.0),
                            "ROC_AUC":          raw.get("roc_auc", 0.0),
                            "NORMAL_Recall":    report.get("NORMAL", {}).get("recall", 0.0),
                            "PNEUMONIA_Recall": report.get("PNEUMONIA", {}).get("recall", 0.0),
                            "PNEUMONIA_F1":     report.get("PNEUMONIA", {}).get("f1-score", 0.0),
                            "Epochs":           raw.get("epochs_total", 0),
                        })
                    except Exception as e:
                        st.error(f"Error loading {filename}: {e}")

    if not data:
        st.warning("No JSON files found in the metrics/ directory.")
        st.stop()

    df = pd.DataFrame(data)
    best_idx = int(df["Accuracy"].idxmax())

    # --- Fejléc metrika kártyák ---
    st.markdown("<div class='section-title'>Overview — best results</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    for col, label, val in [
        (c1, "Best Accuracy",         f"{df['Accuracy'].max()*100:.1f}%"),
        (c2, "Best ROC–AUC",          f"{df['ROC_AUC'].max():.4f}"),
        (c3, "Best NORMAL Recall",    f"{df['NORMAL_Recall'].max()*100:.1f}%"),
        (c4, "Best PNEUMONIA Recall", f"{df['PNEUMONIA_Recall'].max()*100:.1f}%"),
    ]:
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='label'>{label}</div>
                <div class='value'><span>{val}</span></div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Accuracy & ROC–AUC comparison</div>", unsafe_allow_html=True)

    colors_acc = ["#38bdf8" if i == best_idx else "#1e3a5f" for i in range(len(df))]
    colors_auc = ["#0ea5e9" if i == best_idx else "#152a45" for i in range(len(df))]

    fig_acc = go.Figure()
    fig_acc.add_trace(go.Bar(
        name="Accuracy (%)",
        x=df["Model"], y=(df["Accuracy"] * 100).round(2),
        marker_color=colors_acc,
        text=(df["Accuracy"] * 100).round(1).astype(str) + "%",
        textposition="outside",
        textfont=dict(color="#94a3b8", size=11),
    ))
    fig_acc.add_trace(go.Bar(
        name="ROC–AUC × 100",
        x=df["Model"], y=(df["ROC_AUC"] * 100).round(2),
        marker_color=colors_auc,
        text=df["ROC_AUC"].round(4).astype(str),
        textposition="outside",
        textfont=dict(color="#94a3b8", size=11),
    ))
    fig_acc.update_layout(
        **PLOTLY_LAYOUT,
        barmode="group", height=380,
        legend=dict(font=dict(color="#94a3b8"), bgcolor="rgba(0,0,0,0)"),
        yaxis=dict(range=[80, 102], gridcolor="#1e2d4a", linecolor="#1e2d4a", tickfont=dict(color="#94a3b8")),
        xaxis=dict(tickangle=-25, gridcolor="#1e2d4a", linecolor="#1e2d4a", tickfont=dict(color="#94a3b8", size=11)),
    )
    st.plotly_chart(fig_acc, use_container_width=True)

    st.markdown("<div class='section-title'>NORMAL vs PNEUMONIA Recall</div>", unsafe_allow_html=True)

    fig_recall = go.Figure()
    fig_recall.add_trace(go.Bar(
        name="NORMAL Recall",
        x=df["Model"], y=(df["NORMAL_Recall"] * 100).round(2),
        marker_color="#0ea5e9",
        text=(df["NORMAL_Recall"] * 100).round(1).astype(str) + "%",
        textposition="outside",
        textfont=dict(color="#94a3b8", size=11),
    ))
    fig_recall.add_trace(go.Bar(
        name="PNEUMONIA Recall",
        x=df["Model"], y=(df["PNEUMONIA_Recall"] * 100).round(2),
        marker_color="#2563eb",
        text=(df["PNEUMONIA_Recall"] * 100).round(1).astype(str) + "%",
        textposition="outside",
        textfont=dict(color="#94a3b8", size=11),
    ))
    fig_recall.update_layout(
        **PLOTLY_LAYOUT,
        barmode="group", height=380,
        legend=dict(font=dict(color="#94a3b8"), bgcolor="rgba(0,0,0,0)"),
        yaxis=dict(range=[60, 108], gridcolor="#1e2d4a", linecolor="#1e2d4a", tickfont=dict(color="#94a3b8")),
        xaxis=dict(tickangle=-25, gridcolor="#1e2d4a", linecolor="#1e2d4a", tickfont=dict(color="#94a3b8", size=11)),
    )
    st.plotly_chart(fig_recall, use_container_width=True)

    st.markdown("<div class='section-title'>Radar – all models at a glance</div>", unsafe_allow_html=True)

    categories = ["Accuracy", "ROC–AUC", "NORMAL Recall", "PNEUMONIA Recall", "PNEUMONIA F1"]
    palette = ["#38bdf8","#2563eb","#0ea5e9","#7c3aed","#059669","#d97706","#dc2626","#db2777","#84cc16","#14b8a6"]

    fig_radar = go.Figure()
    for i, row in df.iterrows():
        vals = [
            row["Accuracy"] * 100,
            row["ROC_AUC"] * 100,
            row["NORMAL_Recall"] * 100,
            row["PNEUMONIA_Recall"] * 100,
            row["PNEUMONIA_F1"] * 100,
        ]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=categories + [categories[0]],
            name=row["Model"],
            line=dict(color=palette[i % len(palette)], width=2),
            fill="toself",
            fillcolor="rgba(0,0,0,0)",
            opacity=0.9,
        ))
    fig_radar.update_layout(
        **PLOTLY_LAYOUT,
        height=500,
        polar=dict(
            bgcolor="#0f1525",
            radialaxis=dict(visible=True, range=[80, 100], gridcolor="#1e2d4a",
                            tickfont=dict(color="#4a7fa5", size=9), linecolor="#1e2d4a"),
            angularaxis=dict(gridcolor="#1e2d4a", linecolor="#1e2d4a",
                             tickfont=dict(color="#94a3b8", size=11)),
        ),
        legend=dict(font=dict(color="#94a3b8", size=11), bgcolor="rgba(0,0,0,0)", x=1.05),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("<div class='section-title'>Detailed comparison</div>", unsafe_allow_html=True)
    df_display = pd.DataFrame({
        "Model":            df["Model"],
        "Accuracy":         (df["Accuracy"] * 100).round(2).astype(str) + "%",
        "ROC–AUC":          df["ROC_AUC"].round(4),
        "NORMAL Recall":    (df["NORMAL_Recall"] * 100).round(1).astype(str) + "%",
        "PNEUMONIA Recall": (df["PNEUMONIA_Recall"] * 100).round(1).astype(str) + "%",
        "PNEUMONIA F1":     df["PNEUMONIA_F1"].round(4),
        "Epochs":           df["Epochs"],
    })
    st.dataframe(df_display, use_container_width=True, hide_index=True)

elif page == "Real-time Inference":
    st.markdown("<div class='section-title'>X-Ray Image Analysis</div>", unsafe_allow_html=True)

    models_dir = "models"
    available_models = (
        [f.replace(".keras", "") for f in os.listdir(models_dir) if f.endswith(".keras")]
        if os.path.exists(models_dir) else []
    )

    if not available_models:
        st.warning("No .keras files found in the models/ directory.")
    else:
        selected_model_name = st.selectbox("Select a model:", available_models)
        st.info(f"Selected model: **{selected_model_name}**")

    uploaded_file = st.file_uploader("Upload a Chest X-Ray image (JPG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            st.image(image, caption="Uploaded X-Ray", use_container_width=True)

        with col2:
            st.subheader("Prediction")
            if st.button("Run Analysis", type="primary"):
                with st.spinner("Analyzing image..."):
                    model = load_keras_model(selected_model_name)
                    unet_model = None
                    if "unet" in selected_model_name.lower():
                        unet_model = load_unet_model()

                    if model is None:
                        st.error(f"Model '{selected_model_name}.keras' could not be loaded.")
                    else:
                        processed_img, display_img = preprocess_image(
                            image, selected_model_name,
                            target_size=(224, 224),
                            unet_model=unet_model
                        )
                        prediction = model.predict(processed_img)

                        if prediction.shape[-1] == 1:
                            prob = float(prediction[0][0])
                            is_pneumonia = prob > 0.5
                            confidence = prob if is_pneumonia else 1.0 - prob
                        else:
                            is_pneumonia = np.argmax(prediction[0]) == 1
                            confidence = float(np.max(prediction[0]))

                        if is_pneumonia:
                            st.error(f"Prediction: **PNEUMONIA / ABNORMAL**\n\nConfidence: {confidence:.2%}")
                        else:
                            st.success(f"Prediction: **NORMAL**\n\nConfidence: {confidence:.2%}")

                        layer_name = get_last_conv_layer_name(selected_model_name)
                        if layer_name:
                            try:
                                heatmap = make_gradcam_heatmap(processed_img, model, layer_name)
                                cam_image = overlay_gradcam(display_img, heatmap)
                                with col3:
                                    st.image(cam_image, caption="Grad-CAM Heatmap", use_container_width=True)
                                    st.info("Red areas indicate where the model focused to make its decision.")
                            except Exception as e:
                                st.warning(f"Grad-CAM could not be generated: {e}")