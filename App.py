# -*- coding: utf-8 -*-
import streamlit as st
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import io
import pandas as pd
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CarConditionModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        pred = self.classifier(features)
        return pred

@st.cache_resource
def load_model():
    model = CarConditionModel()
    model.load_state_dict(torch.load('car_condition_model.pth', map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def predict_image(image_bytes, model):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        logits = model(image)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred = torch.argmax(logits, dim=1).item()
    
    labels = {0: 'minor', 1: 'moderate', 2: 'severe'}
    return labels[pred], {labels[i]: f"{probs[i]*100:.2f}%" for i in range(3)}

def main():
    st.set_page_config(page_title="Car Condition Classifier", page_icon="🚗", layout="wide")
    st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;  
        padding: 20px;
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stFileUploader {
        border: 2px dashed #4B4BFF;
        border-radius: 10px;
        padding: 20px;
    }
    h1 {
        color: #31333F;
        text-align: center;
        font-family: 'Arial', sans-serif;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }
    .stSuccess {
        background-color: #E6FFE6;
        border-radius: 10px;
        padding: 10px;
    }
    .content-container {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("Car Condition Classifier")
    st.markdown("""
    ### How it works
    - Upload a car photo without license plates.
    - The model will classify the condition as **minor**, **moderate**, or **severe**.
    - Example conditions:
      - **Minor**: Slight scratches or minimal dirt.
      - **Moderate**: Visible damage or moderate dirt.
      - **Severe**: Major damage or heavy dirt.
    """)
    
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    with st.container():
        st.markdown('<div class="content-container">', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Upload Image")
            uploaded_file = st.file_uploader("Drag and drop a photo here or select from a folder", type=["jpg", "jpeg", "png"])
        
        with col2:
            st.subheader("Preview & Result")
            if uploaded_file is not None:
                st.image(uploaded_file, caption="Uploaded photo", width='stretch')
                if st.button("Analyze"):
                    with st.spinner("Analyzing..."):
                        model = load_model()
                        image_bytes = uploaded_file.getvalue()
                        condition, probabilities = predict_image(image_bytes, model)
                        st.success("Analysis result:")
                        st.write(f"**Condition:** {condition}")
                        st.write("**Probabilities:**")
                        for label, prob in probabilities.items():
                            st.write(f"{label}: {prob}")
                        st.session_state.history.append({"file": uploaded_file.name, "condition": condition})
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("### Prediction History")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df)
        csv = df.to_csv(index=False)
        st.download_button("Download History as CSV", csv, "history.csv", "text/csv")

if __name__ == "__main__":
    main()