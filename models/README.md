# CropCare AI – Model Directory

This directory houses the trained deep learning models, classification label mappings, and benchmark evaluation logs for the **CropCare AI** system.

---

## 1. Supported Model Files

The system automatically scans and loads models placed in this directory:

- `baseline_mobilenetv2.keras`: Baseline MobileNetV2 deep learning model (96.97% test accuracy on 27,494 images).
- `optimized_mobilenetv2.keras`: MobileNetV2 model fine-tuned with Particle Swarm Optimization (PSO) (96.73% test accuracy).
- `optimized_mobilenetv2_extended.keras`: Extended training iteration model.

### Supported File Formats:
- Keras v3 Native Archive: `.keras` (Recommended)
- Legacy HDF5: `.h5`
- TensorFlow Lite: `.tflite` (Edge-optimized)
- TensorFlow SavedModel: Directory containing `saved_model.pb` and `variables/`

---

## 2. Class Labels Mapping (`class_names.json`)

The model produces a 17-dimensional softmax probability vector mapping to these exact classes:

```json
[
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_healthy"
]
```

---

## 3. How to Add a New or Updated Model

1. Export your trained model in `.keras` or `.h5` format.
2. Copy the file into this `models/` directory (e.g., `models/my_new_mobilenetv2.keras`).
3. Set the active model path in `config.py`:
   ```python
   DEFAULT_MODEL_FILE = MODELS_DIR / "my_new_mobilenetv2.keras"
   ```
   Or set the environment variable:
   ```powershell
   $env:CROPCARE_MODEL_PATH = "models/my_new_mobilenetv2.keras"
   ```
4. If your model was trained on different class names or ordering, update `models/class_names.json` accordingly.

---

## 4. Input Shape and Preprocessing

- **Input Dimension:** `(224, 224, 3)` (RGB)
- **Normalization:** Standard MobileNetV2 `[-1.0, 1.0]` scaling:
  $$\text{Pixel}_{\text{norm}} = \frac{\text{Pixel}}{127.5} - 1.0$$
