import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import os
from PIL import Image

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

#elofeldolgozas
def preprocess_image(image, model_name, target_size=(224, 224)):
    img = image.resize(target_size)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    name_lower = model_name.lower()

    if "resnet" in name_lower or "densenet" in name_lower:
        img_array = img_array / 255.0
    elif "efficientnet" in name_lower:
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    elif "mobilenet" in name_lower:
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    elif "vgg" in name_lower:
        img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
    return img_array

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
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded X-Ray", use_container_width=True)
            
        with col2:
            st.subheader("Prediction Results")
            if st.button("Run Analysis", type="primary"):
                with st.spinner('Analyzing image...'):
                    model = load_keras_model(selected_model_name)

                    if model is None:
                        st.error(f"Error: Model '{selected_model_name}.keras' could not be loaded")
                    else:
                        processed_img = preprocess_image(image, selected_model_name, target_size=(224, 224))

                        prediction = model.predict(processed_img)

                        if prediction.shape[-1] == 1:
                            #1 kimeneti neuron (sigmoid)
                            prob = float(prediction[0][0])
                            is_pneumonia = prob > 0.5
                            confidence = prob if is_pneumonia else 1.0 - prob
                        else:
                            #2 kimeneti neuron (softmax)
                            is_pneumonia = np.argmax(prediction[0]) == 1
                            confidence = float(np.max(prediction[0]))

                        if is_pneumonia:
                            st.error(f"Prediction: **PNEUMONIA / ABNORMAL** (Confidence: {confidence:.2%})")
                        else:
                            st.success(f"Prediction: **NORMAL** (Confidence: {confidence:.2%})")