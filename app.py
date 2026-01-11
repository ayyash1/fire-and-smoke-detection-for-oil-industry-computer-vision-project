
import streamlit as st
import cv2
import PIL
import numpy as np
from ultralytics import YOLO
import tempfile
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Fire & Smoke Detection System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Industrial Dark Theme ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ff4b4b !important;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Custom Warning/Alert Box */
    .alert-box {
        padding: 15px;
        background-color: #3d0c0c;
        color: #ffcccc;
        border-left: 5px solid #ff4b4b;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #262730;
        color: white;
        border: 1px solid #4a4a4a;
        width: 100%;
    }
    .stButton>button:hover {
        border-color: #ff4b4b;
        color: #ff4b4b;
    }
    
    /* Statistics Text */
    .stat-box {
        text-align: center;
        padding: 10px;
        background-color: #262730;
        border-radius: 5px;
        border: 1px solid #363636;
    }
    .stat-label {
        font-size: 0.8rem;
        color: #a0a0a0;
    }
    .stat-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Helper Functions ---

@st.cache_resource
def load_model(model_path):
    """
    Loads the YOLOv8 model with caching to prevent reloading on every run.
    """
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def process_frame(frame, model, conf_threshold):
    """
    Runs inference on a single frame and returns the annotated frame and detection stats.
    """
    results = model(frame, conf=conf_threshold)
    annotated_frame = results[0].plot() # YOLOv8 built-in plotting
    
    # Extract detection info
    detections = results[0].boxes
    classes = detections.cls.cpu().numpy()
    names = results[0].names
    detected_labels = [names[int(cls)] for cls in classes]
    
    return annotated_frame, detected_labels

# --- Sidebar Controls ---
with st.sidebar:
    st.title("🔥 Fire & Smoke Detection")
    st.markdown("---")
    
    st.header("⚙️ Settings")
    input_source = st.selectbox(
        "Select Input Source",
        ("Image Upload", "Video Upload", "Live Camera (Webcam)")
    )
    
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Adjust detection sensitivity"
    )
    
    st.markdown("---")
    st.markdown("### 📊 System Status")
    status_indicator = st.empty()
    status_indicator.markdown("⚪ **Idle**")

# --- Main Content ---
st.title("🏭 Industrial Safety Monitoring System")
st.markdown("### Real-time Fire & Smoke Detection")

# Load Model
model_path = 'best.pt'
model = load_model(model_path)

if model:
    if input_source == "Image Upload":
        status_indicator.markdown("🟢 **Ready (Image Mode)**")
        
        uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                # Convert file to opencv image
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, 1)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                st.image(image_rgb, use_container_width=True)
            
            with col2:
                st.subheader("Detection Output")
                # Auto-Run Logic
                status_indicator.markdown("🟠 **Processing...**")
                annotated_image, labels = process_frame(image, model, conf_threshold)
                annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                st.image(annotated_image_rgb, use_container_width=True)
                
                # Alert Logic
                if "fire" in [l.lower() for l in labels] or "smoke" in [l.lower() for l in labels]:
                     st.markdown('<div class="alert-box">⚠️ <b>WARNING:</b> Hazard Detected!</div>', unsafe_allow_html=True)
                else:
                    st.success("Analysis Complete: No Hazards Detected.")
                
                status_indicator.markdown("🟢 **Done**")

    elif input_source == "Video Upload":
        status_indicator.markdown("🟢 **Ready (Video Mode)**")
        uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
        
        if uploaded_video is not None:
            # Save to temporary file
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_video.read())
            
            vf = cv2.VideoCapture(tfile.name)
            
            st.markdown("### Video Analysis")
            kpi1, kpi2 = st.columns(2)
            with kpi1:
                st.markdown("**Frame Rate (FPS)**")
                kpi1_text = st.markdown("0")
            with kpi2:
                st.markdown("**Detected Hazards**")
                kpi2_text = st.markdown("None")
                
            stop_button = st.button("Stop Analysis")
            
            st_frame = st.empty()
            
            # Auto-Run Logic
            status_indicator.markdown("🟠 **Running Video Analysis...**")
            
            while vf.isOpened():
                if stop_button:
                    status_indicator.markdown("🔴 **Stopped**")
                    break
                    
                ret, frame = vf.read()
                if not ret:
                    break
                
                # Performance handling
                start_time = time.time()
                annotated_frame, labels = process_frame(frame, model, conf_threshold)
                end_time = time.time()
                fps = 1 / (end_time - start_time)
                
                # Display
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                st_frame.image(annotated_frame_rgb, caption='Analyzing...', use_container_width=True)
                
                # Update Stats
                kpi1_text.write(f"**{fps:.2f}**")
                unique_labels = list(set(labels))
                if unique_labels:
                    kpi2_text.write(f"**{', '.join(unique_labels)}**")
                    if "fire" in [l.lower() for l in unique_labels] or "smoke" in [l.lower() for l in unique_labels]:
                         status_indicator.markdown("🔴 **HAZARD DETECTED**")
                else:
                    kpi2_text.write("None")
                    status_indicator.markdown("🟢 **Monitoring...**")

            vf.release()
            if not stop_button:
                 status_indicator.markdown("🟢 **Analysis Complete**")

    elif input_source == "Live Camera (Webcam)":
        status_indicator.markdown("🟢 **Ready (Webcam Mode)**")
        st.markdown("### Live Real-Time Monitoring")
        
        start_cam = st.button("Start Live Camera", key="start_cam")
        stop_cam = st.button("Stop Live Camera", key="stop_cam")
        
        st_frame = st.empty()
        
        if start_cam:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("Could not open webcam. Ensure no other app is using it.")
            else:
                status_indicator.markdown("🟠 **Monitoring...**")
                
                while cap.isOpened():
                    # Check for stop mechanism - streamlit button state doesn't update inside loop easily
                    # We rely on the "Stop" button triggering a rerun which resets 'start_cam' to False
                    # BUT 'start_cam' button is stateless. It resets to False immediately on next run.
                    # This standard Streamlit pattern is tricky. 
                    # Better Approach: Use Session State to toggle.
                    
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to capture frame.")
                        break
                    
                    # Inference
                    annotated_frame, labels = process_frame(frame, model, conf_threshold)
                    
                    # Display
                    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    st_frame.image(annotated_frame_rgb, caption="Live Feed", use_container_width=True)
                    
                    # Alerts
                    unique_labels = list(set(labels))
                    if "fire" in [l.lower() for l in unique_labels] or "smoke" in [l.lower() for l in unique_labels]:
                         status_indicator.markdown("🔴 **HAZARD DETECTED**")
                    else:
                        status_indicator.markdown("🟢 **Safe**")
                
                cap.release()
        
        # NOTE: The above "button" logic has a flaw in Streamlit: 
        # Click "Start", it runs ONCE. The loop blocks. "Stop" button can't be clicked because the script is busy.
        # FIX: Use a Checkbox for Start/Stop.
        
        # RE-WRITING LOGIC BELOW FOR REPLACEMENT CONTENT
    
    elif input_source == "Live Camera (Webcam)":
        status_indicator.markdown("🟢 **Ready (Webcam Mode)**")
        st.markdown("### Live Real-Time Monitoring")
        
        # Use Toggle/Checkbox for continuous run
        run = st.toggle("Activate Live Camera")
        
        st_frame = st.empty()
        
        if run:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("Could not open webcam. Ensure no other app is using it.")
            else:
                status_indicator.markdown("🟠 **Monitoring...**")
                
                while True:
                    # Streamlit reruns the script when the user interacts with the widget (untoggles).
                    # So we just run until the script is killed/rerun.
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to capture frame.")
                        break
                    
                    annotated_frame, labels = process_frame(frame, model, conf_threshold)
                    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    st_frame.image(annotated_frame_rgb, caption="Live Feed", use_container_width=True)
                    
                    unique_labels = list(set(labels))
                    if "fire" in [l.lower() for l in unique_labels] or "smoke" in [l.lower() for l in unique_labels]:
                         status_indicator.markdown("🔴 **HAZARD DETECTED**")
                    else:
                        status_indicator.markdown("� **Safe**")
                
                cap.release()
        else:
            status_indicator.markdown("⚪ **Idle**")

else:
    st.error("Model not found. Please ensure 'best.pt' is in the directory.")
    
# Footer
st.markdown("---")
st.markdown("🔒 *Industrial Safety Systems v1.0 *")
st.markdown("Project by: Ahamed Ayyash (22-CP-88) & Mohamed Mirzan (22-CP-86)")

