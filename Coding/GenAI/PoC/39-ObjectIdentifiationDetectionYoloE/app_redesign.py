import streamlit as st
import numpy as np
from scipy.ndimage import binary_fill_holes
from ultralytics import YOLOE
from ultralytics.utils.torch_utils import smart_inference_mode
from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor
from huggingface_hub import hf_hub_download
import supervision as sv
import io
from PIL import Image, ImageDraw, ImageFont
from collections import Counter

# Model initialization function (unchanged)
def init_model(model_id, is_pf=False):
    filename = f"{model_id}-seg.pt" if not is_pf else f"{model_id}-seg-pf.pt"
    path = hf_hub_download(repo_id="jameslahm/yoloe", filename=filename)
    model = YOLOE(path)
    model.eval()
    return model

def add_label_info(image, detections):
    # Count occurrences of each class
    class_counts = Counter(detections['class_name'])
    total_detections = len(detections)
    
    # Create a new image with extra space at the top
    new_height = image.height + 100
    new_image = Image.new('RGB', (image.width, new_height), color='white')
    new_image.paste(image, (0, 100))
    
    # Draw label information
    draw = ImageDraw.Draw(new_image)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # Define colors for each class (you can expand this list)
    class_colors = {
        'person': '#FF6B6B',
        'bus': '#4ECDC4',
        'car': '#45B7D1',
        # Add more class-color mappings as needed
    }
    
    # Draw total detections
    draw.text((10, 10), f"Total Detections: {total_detections}", fill='black', font=font)
    
    # Draw individual class counts
    y_offset = 40
    for class_name, count in class_counts.items():
        color = class_colors.get(class_name, '#666666')  # Default to gray if class not in colors
        text = f"{class_name}: {count}"
        draw.rectangle([(10, y_offset-2), (10 + len(text)*10, y_offset+20)], fill=color)
        draw.text((15, y_offset), text, fill='white', font=font)
        y_offset += 30
    
    return new_image

# Inference function (modified for Streamlit)
@smart_inference_mode()
def yoloe_inference(image, prompts, target_image, model_id, image_size, conf_thresh, iou_thresh, prompt_type):
    model = init_model(model_id)
    kwargs = {}
    
    if prompt_type == "Text":
        texts = prompts["texts"]
        model.set_classes(texts, model.get_text_pe(texts))
    # elif prompt_type == "Visual":
    #     kwargs = dict(
    #         prompts=prompts,
    #         predictor=YOLOEVPSegPredictor
    #     )
    #     if target_image:
    #         model.predict(source=image, imgsz=image_size, conf=conf_thresh, iou=iou_thresh, return_vpe=True, **kwargs)
    #         model.set_classes(["object0"], model.predictor.vpe)
    #         model.predictor = None
    #         image = target_image
    #         kwargs = {}
    elif prompt_type == "Prompt-Free":
        vocab = model.get_vocab(prompts["texts"])
        model = init_model(model_id, is_pf=True)
        model.set_vocab(vocab, names=prompts["texts"])
        model.model.model[-1].is_fused = True
        model.model.model[-1].conf = 0.001
        model.model.model[-1].max_det = 1000

    results = model.predict(source=image, imgsz=image_size, conf=conf_thresh, iou=iou_thresh, **kwargs)
    detections = sv.Detections.from_ultralytics(results[0])

    resolution_wh = image.size
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution_wh)

    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(detections['class_name'], detections.confidence)
    ]

    annotated_image = image.copy()
    annotated_image = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX, opacity=0.4).annotate(
        scene=annotated_image, detections=detections)
    annotated_image = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX, thickness=thickness).annotate(
        scene=annotated_image, detections=detections)
    annotated_image = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX, text_scale=text_scale, smart_position=True).annotate(
        scene=annotated_image, detections=detections, labels=labels)

    final_image = annotated_image #add_label_info(annotated_image, detections)

    return final_image



# Main Streamlit app
def main():
    st.set_page_config(page_title="YOLOE Object Detection", layout="wide")
    # st.markdown("""
    #         <style>.stApp {
    #             background: linear-gradient(135deg, lightblue 0%, #8A2BE2 100%);
    #             color: #FFFFFF;
    #         }</style>
    #     """, unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.title("YOLOE: Real-Time Object Detection")
        st.markdown("Powered by Ultralytics | Interactive Detection and Segmentation")
    with col2:              
        st.image(r"H:\Interview Preparation\Coding\GenAI\Tryouts\39-ObjectIdentifiationDetectionYoloE\yoloLogo.jpeg", use_container_width=False)
    
    

    tab1, tab2 = st.tabs(["⚙️ Configuration","⚙️ Object Detection & Segmentation",])

    # Sidebar for controls
    with tab1:
        st.header("Settings")
        
        # Model selection
        model_id = st.selectbox(
            "Model Variant",
            ["yoloe-v8s", "yoloe-v8m", "yoloe-v8l", "yoloe-11s", "yoloe-11m", "yoloe-11l"],
            index=2
        )
        
        # Image parameters
        image_size = st.slider("Image Size", 320, 1280, 640, step=32)
        conf_thresh = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, step=0.05)
        iou_thresh = st.slider("IoU Threshold", 0.0, 1.0, 0.70, step=0.05)
        
        # Prompt mode
        #prompt_mode = st.radio("Prompt Mode", ["Text", "Visual", "Prompt-Free"], index=0)
        prompt_mode = st.radio("Prompt Mode", ["Text", "Prompt-Free"], index=0)

    with tab2:
        st.header("Object Detection & Segmentation")
        # Main content area
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Upload Image")
            
            # Image upload
            uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
            input_image = None
            if uploaded_file:
                input_image = Image.open(uploaded_file)
                st.image(input_image, caption="Uploaded Image", use_container_width =True)

            # Prompt inputs based on mode
            prompts = {}
            target_image = None
            
            if prompt_mode == "Text":
                text_input = st.text_input("Classes (comma-separated)", "person,bus")
                prompts["texts"] = [t.strip() for t in text_input.split(",")]
                
            elif prompt_mode == "Visual":
                visual_type = st.selectbox("Prompt Type", ["Boxes", "Masks"])
                visual_scope = st.selectbox("Scope", ["Same Image", "Different Image"])
                
                if visual_type == "Boxes":
                    # Streamlit doesn't have native box drawing, so we'll use a simple coordinate input
                    st.write("Enter box coordinates (x1,y1,x2,y2)")
                    box_coords = st.text_input("Coordinates", "235,408,342,863")
                    if box_coords:
                        coords = [int(x) for x in box_coords.split(",")]
                        if len(coords) == 4:
                            prompts["bboxes"] = np.array([coords])
                            prompts["cls"] = np.array([0])
                
                elif visual_type == "Masks":
                    mask_file = st.file_uploader("Upload Mask", type=["png"], key="mask")
                    if mask_file and input_image:
                        mask = Image.open(mask_file).convert("L")
                        mask_array = np.array(mask)
                        mask_array = binary_fill_holes(mask_array).astype(np.uint8)
                        mask_array[mask_array > 0] = 1
                        prompts["masks"] = mask_array[None]
                        prompts["cls"] = np.array([0])
                
                if visual_scope == "Different Image":
                    target_file = st.file_uploader("Upload Target Image", type=["jpg", "jpeg", "png"], key="target")
                    if target_file:
                        target_image = Image.open(target_file)
                        st.image(target_image, caption="Target Image", use_container_width =True)
            
            elif prompt_mode == "Prompt-Free":
                # For prompt-free mode, we'll use a default tag list
                try:
                    with open('tools/ram_tag_list.txt', 'r') as f:
                        prompts["texts"] = [x.strip() for x in f.readlines()]
                except:
                    st.warning("Could not load tag list. Using default classes.")
                    prompts["texts"] = ["person", "car", "bus"]

            detect_button = st.button("Detect Objects", type="primary")

        with col2:
            st.subheader("Detection Results")
            if detect_button and input_image:
                with st.spinner("Processing..."):
                    result = yoloe_inference(
                        input_image, prompts, target_image, model_id,
                        image_size, conf_thresh, iou_thresh, prompt_mode
                    )
                    st.image(result, caption="Detected Objects", use_container_width =True)
            else:
                st.info("Upload an image and click 'Detect Objects' to see results")

if __name__ == "__main__":
    main()
