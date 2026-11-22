# CropCare AI – Intelligent Crop Disease Detection System
### IoT-Based Crop Disease Detection System Using Artificial Intelligence
**Deep Learning & IoT in Agriculture**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.47-FF4B4B.svg)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange.svg)](https://tensorflow.org/)
[![Accuracy](https://img.shields.io/badge/Baseline_Accuracy-96.97%25-green.svg)]()
[![Classes](https://img.shields.io/badge/Classes-17-teal.svg)]()

---

## 📌 Executive Summary

**CropCare AI** is an intelligent agricultural decision-support dashboard designed for early detection and classification of foliar crop diseases. Utilizing a transfer-learned **MobileNetV2** deep convolutional neural network enhanced with **Particle Swarm Optimization (PSO)**, the system classifies 17 health states across 5 commercially critical crops from digital leaf imagery.

The project incorporates an **ESP32-CAM microcontroller** edge-acquisition architecture, SQLite historical logging, responsive Plotly analytics, and sustainable agronomic recommendations.

---

## 🌾 Supported Crops & Pathologies (17 Classes)

The system currently classifies foliage across 5 major agricultural crops:

| # | Crop | Scientific Name | Supported Condition / Pathology |
|---|------|-----------------|---------------------------------|
| 1 | **Corn (Maize)** | *Zea mays* | Cercospora Leaf Spot / Gray Leaf Spot |
| 2 | **Corn (Maize)** | *Zea mays* | Common Rust |
| 3 | **Corn (Maize)** | *Zea mays* | Northern Leaf Blight |
| 4 | **Corn (Maize)** | *Zea mays* | Healthy Corn Foliage |
| 5 | **Grape** | *Vitis vinifera* | Black Rot |
| 6 | **Grape** | *Vitis vinifera* | Esca (Black Measles) |
| 7 | **Grape** | *Vitis vinifera* | Leaf Blight (Isariopsis Leaf Spot) |
| 8 | **Grape** | *Vitis vinifera* | Healthy Grape Foliage |
| 9 | **Pepper (Bell)** | *Capsicum annuum* | Bacterial Spot |
| 10 | **Pepper (Bell)** | *Capsicum annuum* | Healthy Bell Pepper Foliage |
| 11 | **Potato** | *Solanum tuberosum* | Early Blight |
| 12 | **Potato** | *Solanum tuberosum* | Late Blight |
| 13 | **Potato** | *Solanum tuberosum* | Healthy Potato Foliage |
| 14 | **Tomato** | *Solanum lycopersicum* | Bacterial Spot |
| 15 | **Tomato** | *Solanum lycopersicum* | Early Blight |
| 16 | **Tomato** | *Solanum lycopersicum* | Late Blight |
| 17 | **Tomato** | *Solanum lycopersicum* | Healthy Tomato Foliage |

---

## 🎯 Model Benchmarks & Quantitative Results

The deep learning model was trained and evaluated on an independent holdout test split of **27,494 images**:

- **Model Architecture:** MobileNetV2 (Inverted Residuals + Depthwise Separable Convolutions)
- **Baseline Test Accuracy:** **96.97%**
- **Baseline Test Loss:** **0.0977**
- **Macro Precision:** **96.61%**
- **Macro Recall:** **96.64%**
- **Macro F1-Score:** **96.60%**
- **PSO Optimized Accuracy:** **96.73%**

> **Academic Presentation Notice:** The 96.97% accuracy reflects offline laboratory benchmarking on the 27,494-image dataset. In live deployment, lighting, blur, and leaf angle will influence real-time confidence scores.

---

## 🏗️ Project Architecture & Folder Structure

```
minor_project/
├── app.py                      # Main entry point & sidebar router
├── config.py                   # Central configuration & class mappings
├── requirements.txt            # Python dependencies
├── README.md                   # Comprehensive project documentation
├── cropcare.db                 # SQLite database for persistent records
│
├── models/                     # Deep learning models & configs
│   ├── baseline_mobilenetv2.keras
│   ├── optimized_mobilenetv2.keras
│   ├── class_names.json        # 17 class labels list
│   ├── model_comparison.json   # Benchmark evaluation metrics
│   └── README.md
│
├── modules/                    # Reusable modular logic
│   ├── __init__.py
│   ├── model_loader.py         # Keras 3 loader, caching & deserialization patch
│   ├── prediction.py           # Inference, top-k ranking & agronomic advisory
│   ├── preprocessing.py        # Image validation, resizing & [-1, 1] normalization
│   ├── database.py             # SQLite CRUD, stats & CSV exporter
│   ├── camera.py               # ESP32-CAM HTTP snapshot acquisition client
│   └── analytics.py            # Plotly interactive charts & benchmarks
│
├── pages/                      # Dashboard UI views
│   ├── __init__.py
│   ├── dashboard.py            # Home page: KPIs, benchmarks, recent scans
│   ├── disease_detection.py    # Main detection page (Upload, Sample, ESP32)
│   ├── history.py              # Filterable history, search & CSV export
│   ├── crop_information.py     # Botany & pathology catalog for 5 crops
│   ├── analytics.py            # Plotly analytics & PSO benchmark comparison
│   └── about.py                # Objectives, system workflow & IoT specs
│
├── assets/                     # Styling & UI resources
│   └── style.css               # Agricultural theme custom stylesheet
│
├── uploads/                    # Stores uploaded and diagnosed leaf photos
└── minor_dataset/              # Offline test dataset images (optional)
```

---

## 🚀 Quick Start Guide (Windows)

### 1. Prerequisites
- **Python 3.10 to 3.13** installed on your system.
- Git (optional).

### 2. Install Dependencies
Open PowerShell or Command Prompt in the project folder and run:
```powershell
pip install -r requirements.txt
```

### 3. Run the Dashboard
Execute the following command:
```powershell
streamlit run app.py
```
Streamlit will launch the dashboard in your default browser at:
```
http://localhost:8501
```

---

## 🧠 Model Configuration & Customization

### Switching Models
In the **Disease Detection** page, use the dropdown to switch between:
1. `MobileNetV2 (Baseline)`
2. `MobileNetV2 (PSO-Optimized)`

### Adding a New Model File
1. Place your `.keras`, `.h5`, or `.tflite` model file inside the `models/` directory.
2. In `config.py`, change:
   ```python
   DEFAULT_MODEL_FILE = MODELS_DIR / "your_new_model.keras"
   ```
   Or set the environment variable:
   ```powershell
   $env:CROPCARE_MODEL_PATH = "models/your_new_model.keras"
   ```
3. Restart the Streamlit app.

### Modifying Class Labels
The class mapping is defined in `models/class_names.json`. The index of each string in this JSON list directly corresponds to the model's output node.

---

## 📷 Future Feature: ESP32-CAM IoT Hardware Integration

The system includes a dedicated `ESP32CamClient` in `modules/camera.py` designed to receive wireless snapshots from an ESP32-CAM module.

### How It Works:
1. **Edge Node:** An ESP32-CAM board connects to the local greenhouse or field Wi-Fi network and runs the standard Arduino `CameraWebServer` firmware.
2. **Snapshot Endpoint:** The camera exposes a snapshot capture URL at `http://<ESP32_IP>/capture`.
3. **Host Processing:** The CropCare AI dashboard sends an HTTP GET request to retrieve the JPEG frame, converts it to RGB, resizes it to 224×224, and executes MobileNetV2 inference locally on the computer.

### Wiring Reference (FTDI to ESP32-CAM for Flashing):
- `5V` (FTDI) ➔ `5V` (ESP32-CAM)
- `GND` (FTDI) ➔ `GND` (ESP32-CAM)
- `TX` (FTDI) ➔ `U0R / RX` (ESP32-CAM)
- `RX` (FTDI) ➔ `U0T / TX` (ESP32-CAM)
- *Note:* Connect `GPIO 0` to `GND` during flashing; disconnect `GPIO 0` and press `RESET` for normal boot.

### Configuring ESP32-CAM in the Dashboard:
1. Go to **Disease Detection** ➔ Select the **📷 ESP32-CAM IoT Hardware** tab.
2. Enter your camera's local IP address (e.g., `192.168.1.50`).
3. Click **📡 Test Camera Connection** to verify reachability.
4. Click **📸 Capture Live Snapshot** to acquire a fresh leaf image and run inference.

---

## 📊 SQLite Database Schema

The database `cropcare.db` automatically initializes the `detections` table:

```sql
CREATE TABLE detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    image_path TEXT,
    crop_name TEXT NOT NULL,
    disease_name TEXT NOT NULL,
    raw_class TEXT NOT NULL,
    confidence REAL NOT NULL,
    health_status TEXT NOT NULL,
    top_predictions_json TEXT,
    model_used TEXT
);
```

---

## 🎓 Academic Presentation Tips

- **Live Demonstration:** Use the **📂 Choose from Test Samples** tab in the Disease Detection page to quickly demonstrate accurate classification without needing to browse files manually.
- **Analytics:** After running 2–3 test scans, navigate to **Analytics** to show the live donut chart and crop distribution updating in real time.
- **Explain PSO:** Particle Swarm Optimization is applied as a population-based metaheuristic technique to optimize hyperparameters and layer weights, balancing exploration and exploitation.
- **Explain MobileNetV2:** Point out that depthwise separable convolutions separate spatial filtering from feature combination, drastically reducing compute load compared to traditional VGG or ResNet models.

---

## 📄 License & Credits
Developed under the title *"IoT-Based Crop Disease Detection System Using Artificial Intelligence"*.
Dataset source: PlantVillage / Open Agricultural Repositories.
