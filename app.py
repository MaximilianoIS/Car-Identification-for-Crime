import os, random, uuid, json, base64
from io import BytesIO

from flask import  Flask, render_template, request, redirect, url_for, flash, session, g, jsonify, current_app
from werkzeug.utils import secure_filename
from types import SimpleNamespace

from PIL import Image, ImageOps

# --- PyTorch, AI Model, and Computer Vision Imports ---
import torch
import torch.nn.functional as F # For softmax
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights # For Car Detection
from sklearn.preprocessing import LabelEncoder
from base_model import BrandThenModelClassifier # Make sure base_model.py is accessible


# --- App and General Configuration ---
UPLOAD_FOLDER = os.path.join('static', 'uploads')
VIDEO_FOLDER = os.path.join('static', 'videos') # For Tab 2
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'} # For Tab 2
SUPPORTED_LANGUAGES = ['en', 'ko']
DEFAULT_LANGUAGE = 'en'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['VIDEO_FOLDER'] = VIDEO_FOLDER # For Tab 2
app.config['SECRET_KEY'] = 'your_very_secret_key_for_sessions_make_this_long_and_random_now_please_and_more_and_even_longer' # Make it strong
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024 # Increased for potential video frames

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VIDEO_FOLDER, exist_ok=True) # Ensure videos folder exists

# --- AI Model Configuration ---
CLASSIFIER_MODEL_PATH = "brand_classifier_best_27.pth" # Your brand/model classifier
# YOLO_MODEL_PATH = "yolov8m.pt" # Not used in this version for car detection
BRAND_CLASSES_PATH = "brand_classes.txt"
MODEL_CLASSES_PATH = "model_classes.txt"
BRAND_MODEL_MAPPING_PATH = "brand_model_mapping.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app.logger.info(f"PyTorch using device: {DEVICE}")

# --- Globals for AI Components (loaded once at startup) ---
brand_classifier_model = None
# object_detector_model = None # Replaced by car_detector_model for video tab
car_detector_model = None
car_detector_weights = None
car_detector_transforms = None
brand_label_encoder = LabelEncoder()
model_label_encoder = LabelEncoder()
brand_model_map = None
image_transform_for_classifier = None


from translations import translations # Make sure you have this file

# --- Language and Translation Setup ---
def get_locale():
    return session.get('language', DEFAULT_LANGUAGE)

@app.before_request
def before_request_tasks():
    g.locale = get_locale()
    g.translations = translations.get(g.locale, translations[DEFAULT_LANGUAGE])
    g._ = lambda key: g.translations.get(key, key)
    g.active_tab = session.get('active_tab', 'image_upload')


def allowed_file(filename, allowed_set=ALLOWED_EXTENSIONS):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_set

# --- AI Component Loading Function ---
def load_ai_components():
    global brand_classifier_model, brand_label_encoder, model_label_encoder, brand_model_map, image_transform_for_classifier
    global car_detector_model, car_detector_weights, car_detector_transforms
    app.logger.info(f"Attempting to load AI components. Device: {DEVICE}")

    try:
        with open(BRAND_CLASSES_PATH, "r", encoding="utf-8") as f: brand_class_list = [line.strip() for line in f]
        brand_label_encoder.fit(brand_class_list)
        app.logger.info(f"Loaded {len(brand_label_encoder.classes_)} brand classes.")

        with open(MODEL_CLASSES_PATH, "r", encoding="utf-8") as f: model_class_list = [line.strip() for line in f]
        model_label_encoder.fit(model_class_list)
        app.logger.info(f"Loaded {len(model_label_encoder.classes_)} model classes.")

        with open(BRAND_MODEL_MAPPING_PATH, "r", encoding="utf-8") as f: brand_model_map = json.load(f)
        app.logger.info("Loaded brand-model mapping.")
    except Exception as e:
        app.logger.error(f"Error loading classifier metadata: {e}", exc_info=True)
        raise

    image_transform_for_classifier = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    app.logger.info("Image transform for classifier defined.")

    try:
        loaded_full_model = torch.load(CLASSIFIER_MODEL_PATH, map_location=DEVICE)
        if isinstance(loaded_full_model, BrandThenModelClassifier): brand_classifier_model = loaded_full_model
        else:
            app.logger.warning("Classifier not full model. Loading state_dict.")
            num_brands = len(brand_label_encoder.classes_); num_models = len(model_label_encoder.classes_)
            brand_classifier_model = BrandThenModelClassifier(num_brands=num_brands, num_models=num_models)
            brand_classifier_model.load_state_dict(loaded_full_model)
        brand_classifier_model.to(DEVICE); brand_classifier_model.eval()
        app.logger.info(f"Brand/Model classifier loaded from {CLASSIFIER_MODEL_PATH}.")
    except Exception as e:
        app.logger.error(f"Error loading Brand/Model classifier: {e}", exc_info=True)
        raise

    try:
        app.logger.info("Loading Car Detector (FasterRCNN ResNet50 FPN)...")
        car_detector_weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        car_detector_model = fasterrcnn_resnet50_fpn(weights=car_detector_weights, progress=True)
        car_detector_model.to(DEVICE)
        car_detector_model.eval()
        car_detector_transforms = car_detector_weights.transforms()
        # Log first few categories to verify, 'car' is typically label 3 (1-indexed)
        if hasattr(car_detector_weights, 'meta') and 'categories' in car_detector_weights.meta:
            app.logger.info(f"Car Detector loaded. COCO Categories (first 5): {car_detector_weights.meta['categories'][:5]}")
        else:
            app.logger.info("Car Detector loaded. (Meta categories not available to log)")

    except Exception as e:
        app.logger.error(f"Error loading Car Detector: {e}", exc_info=True)
        car_detector_model = None # Ensure it's None if loading failed

    app.logger.info("All AI components loaded successfully (or with non-critical errors for car detector).")
    

# --- AI Prediction Function (Classifier) ---
def get_brand_and_model_prediction(pil_image_input):
    global brand_classifier_model, brand_label_encoder, model_label_encoder, brand_model_map, image_transform_for_classifier, DEVICE
    BRAND_CONF_THRESHOLD = 0.40; MODEL_CONF_THRESHOLD = 0.25

    try:
        pil_image = pil_image_input.convert("RGB")
        input_tensor = image_transform_for_classifier(pil_image).unsqueeze(0).to(DEVICE)
        with torch.no_grad(): brand_logits, model_logits = brand_classifier_model(input_tensor)

        brand_probs = F.softmax(brand_logits, dim=1)
        brand_confidence, brand_idx_tensor = torch.max(brand_probs, dim=1)
        brand_idx = brand_idx_tensor.item(); predicted_brand_name = brand_label_encoder.classes_[brand_idx]
        brand_confidence_score = brand_confidence.item()

        app.logger.debug(f"Classifier - Brand: {predicted_brand_name} (Conf: {brand_confidence_score:.2f})")
        if brand_confidence_score < BRAND_CONF_THRESHOLD:
            return {"error": g._('error_low_brand_confidence').format(brand=predicted_brand_name, conf=brand_confidence_score * 100)}

        allowed_models = brand_model_map.get(predicted_brand_name, [])
        if not allowed_models: return {"error": g._('error_no_models_for_brand').format(brand=predicted_brand_name)}
        
        valid_indices, valid_subset = [], []
        for name in allowed_models:
            try: 
                idx = model_label_encoder.transform([name])[0]
                valid_indices.append(idx)
                valid_subset.append(name)
            except ValueError: app.logger.warning(f"Model '{name}' for brand '{predicted_brand_name}' not in model_classes.txt.")
        if not valid_indices: return {"error": g._('error_inconsistent_model_list').format(brand=predicted_brand_name)}

        filtered_logits = model_logits[0, valid_indices]
        if filtered_logits.nelement() == 0: return {"error": g._('error_no_valid_models_for_brand_logits').format(brand=predicted_brand_name)}
        
        model_probs = F.softmax(filtered_logits, dim=0)
        model_conf, local_idx_t = torch.max(model_probs, dim=0)
        predicted_model_name = valid_subset[local_idx_t.item()]; model_confidence_score = model_conf.item()
        app.logger.debug(f"Classifier - Model: {predicted_model_name} (Conf: {model_confidence_score:.2f}) for brand {predicted_brand_name}")

        if model_confidence_score < MODEL_CONF_THRESHOLD:
            return {"brand": predicted_brand_name, "brand_confidence": brand_confidence_score,
                    "error_model": g._('error_low_model_confidence').format(model=predicted_model_name, conf=model_confidence_score * 100)}
        return {"brand": predicted_brand_name, "brand_confidence": brand_confidence_score,
                "model": predicted_model_name, "model_confidence": model_confidence_score}
    except Exception as e:
        app.logger.error(f"Error during classification: {e}", exc_info=True)
        return {"error": g._('error_prediction_failed_general')}

# --- Car Detection Function ---
def detect_cars_on_frame(pil_image, model, transforms, score_thresh=0.6): # Adjusted threshold
    """Detects cars in a PIL image, returns list of [x1, y1, x2, y2] boxes."""
    if model is None or transforms is None:
        app.logger.error("Car detector model or transforms not loaded.")
        return []
    
    img_tensor = transforms(pil_image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(img_tensor)[0]
    
    boxes = []
    # For COCO_V1 weights from torchvision, 'car' is label 3.
    # Verify with weights.meta['categories'] if unsure, it's a list ['__background__', 'person', 'bicycle', 'car', ...]
    car_label_index = 3 
    
    for box, label, score in zip(outputs["boxes"], outputs["labels"], outputs["scores"]):
        if label.item() == car_label_index and score.item() >= score_thresh:
            boxes.append(box.cpu().tolist()) # [x1, y1, x2, y2]
    app.logger.info(f"Car detector found {len(boxes)} cars with score >= {score_thresh}.")
    return boxes

# --- Flask Routes ---
@app.route('/')
def index():
    active_tab = request.args.get('active_tab', session.get('active_tab', 'image_upload'))
    session['active_tab'] = active_tab 
    app.logger.debug(f"Index route: effective active_tab = {active_tab}")
    return render_template('index.html', prediction_data=None, image_url=None, bbox=None, active_tab=active_tab)

@app.route('/set_lang/<lang_code>')
def set_lang(lang_code):
    if lang_code in SUPPORTED_LANGUAGES:
        session['language'] = lang_code
    current_active_tab = session.get('active_tab', 'image_upload')
    return redirect(url_for('index', active_tab=current_active_tab))


# Route for Tab 1: Image Upload and Prediction
@app.route('/predict_image', methods=['POST'])
def predict_image():
    current_active_tab = 'image_upload'
    session['active_tab'] = current_active_tab

    if 'image_file' not in request.files:
        flash(g._('error_no_file_part')); return redirect(url_for('index', active_tab=current_active_tab))
    file = request.files['image_file']
    if file.filename == '':
        flash(g._('error_no_file_selected_flash')); return redirect(url_for('index', active_tab=current_active_tab))

    if file and allowed_file(file.filename):
        original_fn = secure_filename(file.filename)
        unique_fn = f"{uuid.uuid4().hex[:8]}_{original_fn}"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_fn)
        try: 
            pil_image_from_stream = Image.open(file.stream) 
            pil_image_oriented = ImageOps.exif_transpose(pil_image_from_stream) 
            pil_image_oriented.save(image_path) 
            app.logger.info(f"Image saved to {image_path} after potential EXIF transpose.")
        except Exception as e:
            app.logger.error(f"Error processing/saving image: {e}", exc_info=True)
            flash(g._('error_could_not_save').format(error=str(e))); return redirect(url_for('index', active_tab=current_active_tab))
        
        pil_image = Image.open(image_path) 
        processed_image_for_prediction = pil_image.copy() 
        bbox_to_draw_on_original = None 

        crop_coords_str = request.form.get('crop_coords')
        app.logger.info(f"Received crop_coords_str: '{crop_coords_str}'")
        if crop_coords_str:
            try:
                coords = [int(round(float(c.strip()))) for c in crop_coords_str.split(',')]
                app.logger.info(f"Parsed coords from form: {coords}")
                if len(coords) == 4:
                    x1_form, y1_form, x2_form, y2_form = coords
                    w, h = pil_image.size
                    app.logger.info(f"Original image size for crop: {w}x{h}")

                    orig_x1 = min(x1_form, x2_form)
                    orig_y1 = min(y1_form, y2_form)
                    orig_x2 = max(x1_form, x2_form)
                    orig_y2 = max(y1_form, y2_form)

                    final_x1 = max(0, min(orig_x1, w - 1))
                    final_y1 = max(0, min(orig_y1, h - 1))
                    final_x2 = max(final_x1 + 1, min(orig_x2, w)) 
                    final_y2 = max(final_y1 + 1, min(orig_y2, h)) 
                    
                    app.logger.info(f"Clamped crop coords for PIL: x1={final_x1}, y1={final_y1}, x2={final_x2}, y2={final_y2}")

                    if final_x1 < final_x2 and final_y1 < final_y2 :
                        processed_image_for_prediction = pil_image.crop((final_x1, final_y1, final_x2, final_y2))
                        bbox_to_draw_on_original = [final_x1, final_y1, final_x2, final_y2]
                        app.logger.info(f"Successfully cropped to: {bbox_to_draw_on_original}")
                    else:
                        app.logger.warning(f"Invalid clamped crop coordinates: {final_x1},{final_y1},{final_x2},{final_y2}. Using full image.")
                        bbox_to_draw_on_original = None # Important to reset
                        processed_image_for_prediction = pil_image 
                else:
                    app.logger.warning(f"Malformed crop_coords (not 4 values): {crop_coords_str}, using full image.")
                    processed_image_for_prediction = pil_image
            except ValueError as e:
                app.logger.warning(f"Error parsing crop_coords '{crop_coords_str}': {e}, using full image.", exc_info=True)
                processed_image_for_prediction = pil_image
        else:
            app.logger.info("No crop_coords received. Using full image.")
            processed_image_for_prediction = pil_image
        
        prediction_data = get_brand_and_model_prediction(processed_image_for_prediction)
        image_url = url_for('static', filename=f'uploads/{unique_fn}')
        
        return render_template('index.html', prediction_data=prediction_data, image_url=image_url, 
                               bbox=bbox_to_draw_on_original, active_tab=current_active_tab)
    else:
        flash(g._('error_invalid_file_type')); return redirect(url_for('index', active_tab=current_active_tab))

# Route for Tab 2: Get list of videos
@app.route('/get_videos')
def get_videos():
    try:
        videos = [f for f in os.listdir(app.config['VIDEO_FOLDER']) if allowed_file(f, ALLOWED_VIDEO_EXTENSIONS)]
        app.logger.info(f"Found videos in '{app.config['VIDEO_FOLDER']}': {videos}")
        return jsonify(videos) 
    except Exception as e:
        app.logger.error(f"Error listing videos: {e}", exc_info=True)
        return jsonify({"error": g._('error_no_videos_found')}), 500

# ----------------------------------------------------------------------
# /predict_video_frame – Detects cars OR processes a custom crop
# ----------------------------------------------------------------------
@app.route('/predict_video_frame', methods=['POST'])
def predict_video_frame():
    global brand_classifier_model, car_detector_model, car_detector_transforms

    if brand_classifier_model is None:
        current_app.logger.error("Video prediction: brand_classifier_model not loaded.")
        return jsonify({"error": g._("error_video_prediction_failed_model_not_loaded")}), 500

    req = request.get_json(silent=True) or {}
    data_url = req.get("image_data_url")
    if not data_url:
        current_app.logger.warning("predict_video_frame: No image_data_url provided.")
        return jsonify({"error": g._("error_no_image_data_video")}), 400

    try:
        header, b64_data = data_url.split(",", 1)
        pil_frame = Image.open(BytesIO(base64.b64decode(b64_data))).convert("RGB")
    except Exception as exc:
        current_app.logger.error("Frame decode error: %s", exc, exc_info=True)
        return jsonify({"error": g._("error_video_frame_decode")}), 400

    all_predictions = []
    crop_coords_str = req.get("crop_coords")

    if crop_coords_str:
        # --- User provided a specific crop ---
        current_app.logger.info(f"Processing custom crop for video frame: {crop_coords_str}")
        try:
            x1_form, y1_form, x2_form, y2_form = map(int, crop_coords_str.split(","))
            img_w, img_h = pil_frame.size
            
            # Ensure coordinates are valid and within image bounds
            orig_x1 = min(x1_form, x2_form)
            orig_y1 = min(y1_form, y2_form)
            orig_x2 = max(x1_form, x2_form)
            orig_y2 = max(y1_form, y2_form)

            final_x1 = max(0, min(orig_x1, img_w - 1))
            final_y1 = max(0, min(orig_y1, img_h - 1))
            final_x2 = max(final_x1 + 1, min(orig_x2, img_w)) 
            final_y2 = max(final_y1 + 1, min(orig_y2, img_h)) 

            if final_x1 < final_x2 and final_y1 < final_y2:
                cropped_image = pil_frame.crop((final_x1, final_y1, final_x2, final_y2))
                bbox_for_response = [final_x1, final_y1, final_x2, final_y2]
                classification_result = get_brand_and_model_prediction(cropped_image)
                prediction_item = {
                    "label": g._("label_custom_area"), # e.g., "Custom Area"
                    "bbox": bbox_for_response,
                    "classification": classification_result
                }
                all_predictions.append(prediction_item)
                current_app.logger.info(f"Custom crop processed: {bbox_for_response}")
            else:
                current_app.logger.warning(f"Invalid custom crop coordinates after clamping: {crop_coords_str}. No prediction made.")
                # Optionally return an error message for this case
                all_predictions.append({
                    "label": g._("label_custom_area"),
                    "bbox": [0,0,pil_frame.width, pil_frame.height], # Fallback bbox
                    "classification": {"error": g._("error_invalid_crop_video")}
                })

        except ValueError:
            current_app.logger.warning(f"Malformed crop_coords for video frame: '{crop_coords_str}'. No prediction made.")
            all_predictions.append({
                "label": g._("label_custom_area"),
                "bbox": [0,0,pil_frame.width, pil_frame.height],
                "classification": {"error": g._("error_malformed_crop_video")}
            })
    else:
        # --- No crop_coords, so detect all cars ---
        current_app.logger.info("No custom crop. Detecting cars in video frame.")
        if car_detector_model is None:
            current_app.logger.error("Car detector model not loaded, cannot detect cars.")
            return jsonify({"error": g._("error_car_detector_not_loaded")}), 500

        detected_car_boxes = detect_cars_on_frame(pil_frame, car_detector_model, car_detector_transforms)
        
        if not detected_car_boxes:
            current_app.logger.info("No cars detected in the frame.")
            all_predictions.append({
                "label": g._("label_no_cars_detected"), # e.g., "No Cars Detected"
                "bbox": [0, 0, pil_frame.width, pil_frame.height], # Represents full frame
                "classification": {"error": g._("error_no_cars_found_in_frame")}
            })
        else:
            for i, car_bbox in enumerate(detected_car_boxes):
                # Ensure car_bbox components are integers for cropping
                x1, y1, x2, y2 = map(int, car_bbox)
                
                # Validate and clamp car_bbox coordinates to be safe for PIL crop
                img_w, img_h = pil_frame.size
                crop_x1 = max(0, x1)
                crop_y1 = max(0, y1)
                crop_x2 = min(img_w, x2) # Ensure x2 is not beyond image width
                crop_y2 = min(img_h, y2) # Ensure y2 is not beyond image height

                if crop_x1 >= crop_x2 or crop_y1 >= crop_y2: # Check if the box is valid
                    current_app.logger.warning(f"Skipping invalid car_bbox after clamping: original {car_bbox}, clamped [{crop_x1},{crop_y1},{crop_x2},{crop_y2}]")
                    continue
                
                car_image_crop = pil_frame.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                classification_result = get_brand_and_model_prediction(car_image_crop)
                
                prediction_item = {
                    "label": f"{g._('label_car_prefix')} {i+1}", # e.g., "Car 1"
                    "bbox": [crop_x1, crop_y1, crop_x2, crop_y2], # Bbox relative to original frame
                    "classification": classification_result
                }
                all_predictions.append(prediction_item)
            current_app.logger.info(f"Processed {len(all_predictions)} detected cars.")

    return jsonify({"predictions": all_predictions}), 200

    
# --- Main Execution ---
if __name__ == '__main__':
    with app.app_context(): 
        try: 
            load_ai_components()
        except Exception as e_load:
            print(f"CRITICAL STARTUP ERROR: AI components failed to load: {e_load}")
            import traceback
            traceback.print_exc()
            # Consider sys.exit(1) for production
    
    app.run(debug=True, port=5001)