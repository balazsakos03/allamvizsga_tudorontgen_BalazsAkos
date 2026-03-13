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
def load_keras_model(model_path):
    # return tf.keras.models.load_model(model_path)
    return "Model loaded from: " + model_path

#elofeldolgozas
def preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
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
    
    available_models = ["ResNet50", "DenseNet121", "EfficientNetB0", "MobileNetV2", "VGG16", "VGG16_custom_threshold"]
    selected_model_name = st.selectbox("Select a model for inference:", available_models)
    st.info(f"Currently selected: **{selected_model_name}**")
    
    uploaded_file = st.file_uploader("Upload a Chest X-Ray image (JPG, PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded X-Ray", use_column_width=True)
            
        with col2:
            st.subheader("Prediction Results")
            if st.button("Run Analysis", type="primary"):
                with st.spinner('Analyzing image...'):
                    #szimulalt eredmeny
                    simulated_prediction = np.random.rand() 
                    if simulated_prediction > 0.5:
                        st.error(f"Prediction: **Pneumonia / Abnormal** (Confidence: {simulated_prediction:.2%})")
                    else:
                        st.success(f"Prediction: **Normal** (Confidence: {1 - simulated_prediction:.2%})")