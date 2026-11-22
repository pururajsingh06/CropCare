"""
CropCare AI - Configuration Module
Central settings for model paths, image preprocessing, database, and system parameters.
"""

from pathlib import Path
import os

# Base Directories
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"
UPLOADS_DIR = BASE_DIR / "uploads"
DATASET_DIR = BASE_DIR / "minor_dataset"
DEMO_SAMPLES_DIR = BASE_DIR / "demo_samples"

# Ensure runtime directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Database Configuration
DATABASE_PATH = BASE_DIR / "cropcare.db"

# Model Configuration
DEFAULT_MODEL_FILE = os.getenv("CROPCARE_MODEL_PATH", str(MODELS_DIR / "baseline_mobilenetv2.keras"))
PSO_MODEL_FILE = os.getenv("CROPCARE_PSO_MODEL_PATH", str(MODELS_DIR / "optimized_mobilenetv2.keras"))
CLASS_NAMES_FILE = os.getenv("CROPCARE_CLASS_NAMES_PATH", str(MODELS_DIR / "class_names.json"))
MODEL_COMPARISON_FILE = str(MODELS_DIR / "model_comparison.json")

# Model Input Specs
IMAGE_INPUT_SHAPE = (224, 224)
IMAGE_INPUT_CHANNELS = 3
TARGET_SIZE = (224, 224)

# Preprocessing Mode:
# "mobilenet_v2": (x / 127.5) - 1.0 (standard MobileNetV2 normalization [-1, 1])
# "rescale_255": x / 255.0 ([0, 1])
# "raw": x (no scaling)
DEFAULT_PREPROCESSING = "mobilenet_v2"

# Supported Image Formats
SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png", "webp"]

# Image & Out-of-Distribution (OOD) Validation Thresholds
MIN_CONFIDENCE_THRESHOLD = 80.0       # Minimum % confidence required to accept prediction (filters out non-target crops like Money Plant)
MIN_PREDICTION_MARGIN = 50.0          # Minimum % margin between top-1 and top-2 candidate
MAX_PREDICTION_ENTROPY = 0.65         # Maximum Shannon entropy before flagging OOD/uncertainty
MIN_FOLIAGE_COVERAGE = 0.25           # Minimum foliage color ratio (25%) required
MIN_GREEN_TISSUE_RATIO = 0.25         # Minimum chlorophyll green leaf tissue ratio (25%) required
MIN_IMAGE_VARIANCE = 15.0             # Minimum standard deviation of pixel values (detects blank/solid)
MAX_UNNATURAL_COLOR_RATIO = 0.18      # Max artificial/unnatural color ratio (blue/cyan/magenta)

# ESP32-CAM Configuration (IoT Hardware)
DEFAULT_CAMERA_IP = os.getenv("ESP32_CAM_IP", "192.168.1.50")
DEFAULT_CAMERA_PORT = 80
CAMERA_CAPTURE_PATH = "/capture"
CAMERA_STREAM_PATH = "/stream"
CAMERA_TIMEOUT = 6  # seconds

# Dataset & Benchmark Statistics (Verified Ground Truth)
BENCHMARK_STATS = {
    "supported_crops_count": 5,
    "total_classes_count": 17,
    "model_architecture": "MobileNetV2",
    "baseline_test_accuracy": 96.97,
    "baseline_test_loss": 0.0977,
    "baseline_macro_precision": 96.61,
    "baseline_macro_recall": 96.64,
    "baseline_macro_f1": 96.60,
    "pso_test_accuracy": 96.73,
    "pso_test_loss": 0.1000,
    "pso_macro_precision": 96.54,
    "pso_macro_recall": 96.04,
    "pso_macro_f1": 96.26,
    "total_training_images": 27494,
    "optimization_technique": "Particle Swarm Optimization (PSO)",
}

# The 5 Supported Crops Metadata
SUPPORTED_CROPS = {
    "Corn": {
        "name": "Corn (Maize)",
        "scientific_name": "Zea mays",
        "icon": "🌽",
        "category": "Cereal Grain",
        "description": "One of the most widely cultivated cereal crops globally, essential for food, feed, and biofuel production.",
        "common_diseases": [
            "Cercospora Leaf Spot / Gray Leaf Spot",
            "Common Rust",
            "Northern Leaf Blight"
        ],
        "growth_conditions": "Thrives in warm, sunny weather (20°C–30°C) with well-drained, nutrient-rich fertile soil."
    },
    "Grape": {
        "name": "Grape",
        "scientific_name": "Vitis vinifera",
        "icon": "🍇",
        "category": "Fruit / Viticulture",
        "description": "A commercially vital woody perennial vine widely grown for table grapes, wine production, and raisins.",
        "common_diseases": [
            "Black Rot",
            "Esca (Black Measles)",
            "Leaf Blight (Isariopsis Leaf Spot)"
        ],
        "growth_conditions": "Prefers Mediterranean and temperate climates with ample sunlight and well-aerated sandy loam."
    },
    "Pepper": {
        "name": "Bell Pepper",
        "scientific_name": "Capsicum annuum",
        "icon": "🫑",
        "category": "Vegetable / Solanaceae",
        "description": "High-value horticultural crop rich in Vitamin C, susceptible to bacterial leaf spots in warm, humid weather.",
        "common_diseases": [
            "Bacterial Spot"
        ],
        "growth_conditions": "Requires warm temperatures (21°C–29°C), consistent moderate moisture, and loose loam soil."
    },
    "Potato": {
        "name": "Potato",
        "scientific_name": "Solanum tuberosum",
        "icon": "🥔",
        "category": "Tuber / Solanaceae",
        "description": "The world's fourth most important food crop, highly susceptible to devastating fungal and water mold blights.",
        "common_diseases": [
            "Early Blight",
            "Late Blight"
        ],
        "growth_conditions": "Prefers cool to temperate climates (15°C–20°C), slightly acidic well-drained soil, and moderate irrigation."
    },
    "Tomato": {
        "name": "Tomato",
        "scientific_name": "Solanum lycopersicum",
        "icon": "🍅",
        "category": "Fruit / Solanaceae",
        "description": "Globally cultivated dietary staple prone to bacterial spots and foliage blights in humid microclimates.",
        "common_diseases": [
            "Bacterial Spot",
            "Early Blight",
            "Late Blight"
        ],
        "growth_conditions": "Needs warm sunshine (20°C–27°C), consistent watering, and nutrient-dense, loam soil."
    }
}

# Mapping helper from raw dataset class string to friendly parsed entities
def parse_class_label(raw_label: str) -> dict:
    """
    Parses a raw class name from the dataset (e.g. 'Corn_(maize)___Common_rust_')
    into standardized, human-readable metadata.
    """
    is_healthy = "healthy" in raw_label.lower()
    
    # Identify crop
    crop = "Unknown"
    lower = raw_label.lower()
    if "corn" in lower:
        crop = "Corn (Maize)"
    elif "grape" in lower:
        crop = "Grape"
    elif "pepper" in lower:
        crop = "Bell Pepper"
    elif "potato" in lower:
        crop = "Potato"
    elif "tomato" in lower:
        crop = "Tomato"
        
    # Clean disease name
    if is_healthy:
        disease = "Healthy Leaf"
        status = "Healthy"
    else:
        status = "Diseased"
        # Extract disease portion
        if "___" in raw_label:
            disease_part = raw_label.split("___", 1)[1]
        elif "_" in raw_label:
            # e.g., Tomato_Bacterial_spot
            disease_part = "_".join(raw_label.split("_")[1:])
        else:
            disease_part = raw_label
            
        disease = disease_part.replace("_", " ").strip()
        # Clean double spaces
        disease = " ".join(disease.split())
        
        # Specific formatting cleanups
        if "cercospora" in disease.lower():
            disease = "Cercospora Leaf Spot / Gray Leaf Spot"
        elif "common rust" in disease.lower():
            disease = "Common Rust"
        elif "northern leaf blight" in disease.lower():
            disease = "Northern Leaf Blight"
        elif "black rot" in disease.lower():
            disease = "Black Rot"
        elif "esca" in disease.lower():
            disease = "Esca (Black Measles)"
        elif "isariopsis" in disease.lower() or "leaf blight" in disease.lower():
            disease = "Leaf Blight (Isariopsis Spot)"
        elif "bacterial spot" in disease.lower():
            disease = "Bacterial Spot"
        elif "early blight" in disease.lower():
            disease = "Early Blight"
        elif "late blight" in disease.lower():
            disease = "Late Blight"

    return {
        "raw_label": raw_label,
        "crop": crop,
        "disease": disease,
        "status": status,
        "is_healthy": is_healthy
    }
