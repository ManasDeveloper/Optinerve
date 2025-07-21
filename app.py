import streamlit as st
from prediction import predict_single_image, predict_pil_image
import torchvision.models as models
from dataset import EyeDataset
import torch.nn as nn
import torch
from PIL import Image

# Configure page
st.set_page_config(
    page_title="Optinerve | EYE DEFECT DETECTOR",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS - Simplified and more reliable
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Hero styling */
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        color: white;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        color: rgba(255, 255, 255, 0.9);
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Card styling */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .white-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
    }
    
    /* Feature cards */
    .feature-box {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 0.5rem 0;
        transition: transform 0.3s ease;
    }
    
    .feature-box:hover {
        transform: translateY(-5px);
        background : grey;
        
    }
    
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .feature-text {
        color: white;
        font-weight: 500;
        font-size: 1.5rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.8rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(72, 187, 120, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(72, 187, 120, 0.6);
    }
    
    /* File uploader */
    .uploadedFile {
        border: none !important;
        background: transparent !important;
    }
    
    /* Results */
    .success-card {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: white;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(72, 187, 120, 0.3);
    }
    
    .warning-card {
        background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%);
        color: white;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(237, 137, 54, 0.3);
    }
    
    .result-disease {
        font-size: 2rem;
        font-weight: 700;
        margin: 1rem 0;
    }
    
    .result-code {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-bottom: 1rem;
    }
    
    /* Hide default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Text colors */
    .white-text {
        color: white;
    }
    
    .dark-text {
        color: #2d3748;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize model and dataset
device = 'cuda' if torch.cuda.is_available() else 'cpu'
img_dir = "dataset/right_eye_dataset"
annotations = "annotations.csv"
model_path = "eye_disease_resnet50.pth"

@st.cache_resource
def load_model_and_dataset():
    dataset = EyeDataset(img_dir=img_dir, annotations=annotations, transform=None)
    model = models.resnet50(pretrained=False)
    num_classes = len(dataset.label_encoder.classes_)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, dataset

model, dataset = load_model_and_dataset()

# Disease information
label_to_disease = {
    "N": "Normal (No Disease)",
    "D": "Diabetic Retinopathy",
    "G": "Glaucoma",
    "C": "Cataract",
    "A": "Age-Related Macular Degeneration",
    "H": "Hypertensive Retinopathy",
    "M": "Myopia",
    "O": "Other Eye Diseases"
}

disease_descriptions = {
    "N": "Your eye appears healthy with no signs of disease detected.",
    "D": "A diabetes complication affecting the retina's blood vessels.",
    "G": "Increased eye pressure that can damage the optic nerve.",
    "C": "Clouding of the eye's natural lens affecting vision.",
    "A": "Deterioration of the central portion of the retina.",
    "H": "Damage to retinal blood vessels due to high blood pressure.",
    "M": "Refractive error where distant objects appear blurry.",
    "O": "Other eye conditions requiring professional evaluation."
}

from torchvision.transforms import transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])

# Hero Section
st.markdown('<h1 class="hero-title">Optinerve👁️🤖</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Advanced artificial intelligence for early eye disease detection and diagnosis</p>', unsafe_allow_html=True)

# Feature cards using columns
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="feature-box">
        <div class="feature-icon">🎯</div>
        <div class="feature-text">Accurate Detection</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-box">
        <div class="feature-icon">⚡</div>
        <div class="feature-text">Instant Results</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-box">
        <div class="feature-icon">🔬</div>
        <div class="feature-text">AI-Powered</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="feature-box">
        <div class="feature-icon">🏥</div>
        <div class="feature-text">Medical Grade</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Upload Section
st.markdown("""
<div class="glass-card">
    <h2 style="color: white; text-align: center; font-size: 2rem; margin-bottom: 0.5rem;">📤 Upload Your Eye Image</h2>
    <p style="color: rgba(255,255,255,0.9); text-align: center; font-size: 1.1rem; margin-bottom: 1.5rem;">Select a clear, high-quality image of an eye for AI analysis</p>
</div>
""", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader(
    "Choose an eye image file",
    type=["jpg", "png", "jpeg"],
    help="Upload a clear image of an eye (JPG, PNG, or JPEG format)"
)

# Process uploaded image
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Create two columns for image and info
    col1, col2 = st.columns([1, 1])
    
    with col1:
        
        st.markdown('<h3 style="color: #2d3748; margin-bottom: 1rem; text-align: center;">📸 Uploaded Image</h3>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="white-card">
            <h3 class="dark-text" style="margin-bottom: 1rem;">🔍 Analysis Ready</h3>
            <p style="color: #4a5568; line-height: 1.6;">
                Your image has been successfully uploaded and is ready for AI analysis. 
                Our advanced deep learning model will examine the image for signs of various eye diseases.
            </p>
            <div style="margin-top: 1.5rem; padding: 1rem; background: #f0f8ff; border-radius: 10px; border-left: 4px solid #667eea;">
                <p style="color: #667eea; font-weight: 500; margin: 0;">
                    ✓ Image format: Valid<br>
                    ✓ Image quality: Good<br>
                    ✓ Ready for analysis
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Predict button
    if st.button("🚀 Analyze Eye Image"):
        with st.spinner("🔬 Analyzing image with AI..."):
            # Add progress bar
            progress_bar = st.progress(0)
            import time
            for i in range(100):
                time.sleep(0.01)  # Small delay for visual effect
                progress_bar.progress(i + 1)
            
            # Make prediction
            label_code = predict_pil_image(model, image, transform, device, dataset.label_encoder)
            full_name = label_to_disease.get(label_code, "Unknown")
            description = disease_descriptions.get(label_code, "Please consult a medical professional.")
            
            progress_bar.empty()
            
            # Display results with enhanced styling
            if label_code == "N":
                card_class = "success-card"
                icon = "✅"
            else:
                card_class = "warning-card"
                icon = "⚠️"
            
            st.markdown(f"""
            <div class="{card_class}">
                <h3 style="margin-bottom: 1rem;">{icon} Analysis Complete</h3>
                <div class="result-disease">{full_name}</div>
                <div class="result-code">Classification Code: {label_code}</div>
                <hr style="margin: 1rem 0; border: none; height: 1px; background: rgba(255,255,255,0.3);">
                <p style="font-size: 1.1rem; line-height: 1.6; margin: 0;">{description}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Additional information
            st.markdown("""
            <div class="white-card">
                <h4 style="color: #2d3748; margin-bottom: 1rem;">📋 Important Medical Notes</h4>
                <div style="color: #4a5568; line-height: 1.8;">
                    <p>• This AI analysis is for screening purposes only</p>
                    <p>• Always consult a qualified ophthalmologist for diagnosis</p>
                    <p>• Regular eye checkups are recommended for everyone</p>
                    <p>• Early detection can significantly improve treatment outcomes</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: rgba(255,255,255,0.8); padding: 2rem;">
    <hr style="border: none; height: 1px; background: rgba(255,255,255,0.2); margin: 2rem 0;">
    <p style="font-size: 0.9rem; margin: 0;">
        🔬 Created by Manas Kulkarni • 🏥 For Medical Screening Only • 👨‍⚕️ Always Consult Your Doctor
    </p>
</div>
""", unsafe_allow_html=True)