import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import os
import cv2
from PIL import Image
import matplotlib.cm as cm

#page config
st.set_page_config(page_title="Chest X-Ray Classifier", layout="wide")

#modell betoltese
@st.cache_resource
def load_keras_model(model_name):
    model_path = os.path.join("models", f"{model_name}.keras")
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        None

@st.cache_resource
def load_unet_model():
    #unet model betoltese
    unet_path = os.path.join("models", "unet_lung_seg.hdf5")
    if not os.path.exists(unet_path):
        unet_path = os.path.join("models", "unet_model.keras") 
        
    if os.path.exists(unet_path):
        return tf.keras.models.load_model(unet_path, compile=False) 
    return None

#elofeldolgozas
def preprocess_image(image, model_name, target_size=(224, 224), unet_model=None):
    img_array = np.array(image)
    name_lower = model_name.lower()
    
    #unet szegmentalas es auto crop
    if "unet" in name_lower and unet_model is not None:
        img_uint8 = np.clip(img_array, 0, 255).astype(np.uint8)
        
        #unet bemeneti meretek
        unet_input_shape = unet_model.input_shape
        unet_img_size = (unet_input_shape[1], unet_input_shape[2])
        unet_channels = unet_input_shape[3]
        
        #kep elokeszitese az unet szamara
        img_unet = cv2.resize(img_uint8, unet_img_size)
        img_unet = img_unet / 255.0
        
        if unet_channels == 1:
            img_unet_gray = cv2.cvtColor((img_unet * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            img_unet = img_unet_gray / 255.0
            img_unet = np.expand_dims(img_unet, axis=-1)
            
        img_tensor = np.expand_dims(img_unet, axis=0).astype(np.float32)
        
        #maszkolas
        mask = unet_model(img_tensor, training=False)[0].numpy()
        mask_resized = cv2.resize(mask, (img_array.shape[1], img_array.shape[0]))
        
        #binarizalas es biztos 2d forma
        if len(mask_resized.shape) == 3:
            mask_resized = np.squeeze(mask_resized)
        mask_binary = (mask_resized > 0.5).astype(np.uint8)
        
        #maszk raszorzas
        masked_img = cv2.bitwise_and(img_uint8, img_uint8, mask=mask_binary)
        
        #auto crop
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            pad = 10
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(img_array.shape[1], x + w + pad)
            y2 = min(img_array.shape[0], y + h + pad)
            masked_img = masked_img[y1:y2, x1:x2]
            
        img_array = cv2.resize(masked_img, target_size)
    else:
        img_resized = image.resize(target_size)
        img_array = np.array(img_resized)

    #clahe
    if "clahe" in name_lower:
        img_uint8 = np.clip(img_array, 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        img_array = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    display_image = Image.fromarray(np.uint8(img_array))

    #modell specifikus elofeldolgozas
    img_array = img_array.astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0) # (1, 224, 224, 3)
    
    if "resnet" in name_lower or "densenet" in name_lower:
        img_array = img_array / 255.0
    elif "efficientnet" in name_lower:
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    elif "mobilenet" in name_lower:
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    elif "vgg" in name_lower:
        img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
        
    return img_array, display_image

#grad cam vizualizacio
def get_last_conv_layer_name(model_name):
    name_lower = model_name.lower()
    if "resnet" in name_lower: return "conv5_block3_out"
    elif "densenet" in name_lower: return "relu"
    elif "efficientnet" in name_lower: return "top_activation"
    elif "mobilenet" in name_lower: return "out_relu"
    elif "vgg" in name_lower: return "block5_conv3"
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
        
        #sigmoid vagy softmax
        if preds.shape[-1] == 1:
            class_channel = preds[0][0]
        else:
            pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

    #gradiens szamitas
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    #hoterkep sulyozasa
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    #normalizalas 0 es 1 koze
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_gradcam(img, heatmap, alpha = 0.4):
    img_array = np.array(img)
    heatmap = np.uint8(255 * heatmap)

    #jet colormap szinezeshez
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    #hoterkep meretezese az eredeti kep alapjan
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img_array.shape[1], img_array.shape[0]))
    jet_heatmap = np.array(jet_heatmap)

    #ravetites az eredeti kepre
    superimposed_img = jet_heatmap * alpha + img_array
    superimposed_img = tf.keras.utils.array_to_img(superimposed_img)
    return superimposed_img

#ui es navigacio
st.title("Chest X-Ray Classification & Analysis 🫁")
st.sidebar.title("Navigation")

page = st.sidebar.radio("Go to:", ["Model Statistics", "Real-time Inference"])

#statisztika oldal
if page == "Model Statistics":
    st.header("Model Performance Comparison")
    st.write("Comparing the evaluation metrics of the trained deep learning models.")
    
    metrics_dir = "metrics"
    
    if os.path.exists(metrics_dir) and len(os.listdir(metrics_dir)) > 0:
        data = []
        
        for filename in os.listdir(metrics_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(metrics_dir, filename)
                with open(filepath, 'r') as f:
                    try:
                        raw_data = json.load(f)
                        
                        #json adatok kinyerese
                        parsed_data = {
                            "Model": raw_data.get("model", "Unknown"),
                            "Accuracy": raw_data.get("classification_report", {}).get("accuracy", 0.0),
                            "ROC AUC": raw_data.get("roc_auc", 0.0),
                            "Pneumonia F1-Score": raw_data.get("classification_report", {}).get("PNEUMONIA", {}).get("f1-score", 0.0),
                            "Epochs": raw_data.get("epochs_total", 0),
                            "Batch Size": raw_data.get("batch_size", 0)
                        }
                        data.append(parsed_data)
                    except Exception as e:
                        st.error(f"Error loading {filename}: {e}")
        
        if data:
            df = pd.DataFrame(data)
            
            #formazas a tablazatos megjeleniteshez
            df_display = df.copy()
            df_display["Accuracy"] = (df_display["Accuracy"] * 100).round(2).astype(str) + "%"
            df_display["ROC AUC"] = df_display["ROC AUC"].round(4)
            df_display["Pneumonia F1-Score"] = df_display["Pneumonia F1-Score"].round(4)
            
            st.subheader("Metrics Summary Table")
            st.dataframe(df_display, use_container_width=True)
            
            #grafikonok megjelenitese
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Accuracy Comparison")
                chart_data_acc = df.set_index("Model")[["Accuracy"]]
                st.bar_chart(chart_data_acc)
                
            with col2:
                st.subheader("ROC AUC Comparison")
                chart_data_roc = df.set_index("Model")[["ROC AUC"]]
                st.bar_chart(chart_data_roc)
                
    else:
        st.warning("No JSON files found in the 'metrics' folder. Please add your files there.")

#predikcio oldal
elif page == "Real-time Inference":
    st.header("X-Ray Image Analysis")
    
    models_dir = "models"
    if os.path.exists(models_dir):
        available_models = [f.replace(".keras", "") for f in os.listdir(models_dir) if f.endswith(".keras")]
    else:
        available_models = []

    if not available_models:
        st.warning("No .keras files found in the 'models' directory.")
    else:
        selected_model_name = st.selectbox("Select a model for inference:", available_models)
        st.info(f"Currently selected: **{selected_model_name}**")
    
    uploaded_file = st.file_uploader("Upload a Chest X-Ray image (JPG, PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.image(image, caption="Uploaded X-Ray", use_container_width=True)
            
        with col2:
            st.subheader("Prediction Results")
            if st.button("Run Analysis", type="primary"):
                with st.spinner('Analyzing image...'):
                    model = load_keras_model(selected_model_name)

                    unet_model = None
                    if "unet" in selected_model_name.lower():
                        unet_model = load_unet_model()
                    
                    if model is None:
                        st.error(f"Error: Model '{selected_model_name}.keras' could not be loaded.")
                    else:
                        processed_img, display_img = preprocess_image(image, selected_model_name, target_size=(224, 224), unet_model=unet_model)
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
                            
                        #grad-cam generalas
                        layer_name = get_last_conv_layer_name(selected_model_name)
                        if layer_name:
                            try:
                                heatmap = make_gradcam_heatmap(processed_img, model, layer_name)
                                cam_image = overlay_gradcam(display_img, heatmap)
                                
                                #megjelenites a harmadik oszlopban
                                with col3:
                                    st.image(cam_image, caption="Grad-CAM Heatmap", use_container_width=True)
                                    st.info("💡 The heatmap shows where the model focused to make its decision. Red areas are the most important features.")
                            except Exception as e:
                                st.warning(f"Grad-CAM could not be generated: {e}")