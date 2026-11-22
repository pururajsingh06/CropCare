"""
CropCare AI - Model Loader Module
Provides flexible, robust loading for MobileNetV2 deep learning models
(.keras, .h5, SavedModel, .tflite) with Keras 3 compatibility patches and caching.
"""

from typing import Tuple, List, Optional, Any, Dict
from pathlib import Path
import json
import os
import streamlit as st
import numpy as np

# Apply Keras 3 quantization_config deserialization patch before loading models
def _patch_keras_deserialization():
    """
    Patches Keras 3 layer deserializers to safely ignore 'quantization_config'
    when models saved in newer Keras 3 minor versions are loaded in earlier runtimes.
    """
    try:
        import keras
        for layer_name in ["Dense", "Conv2D", "DepthwiseConv2D", "BatchNormalization"]:
            if hasattr(keras.layers, layer_name):
                layer_cls = getattr(keras.layers, layer_name)
                orig_from_config = getattr(layer_cls, "from_config", None)
                if orig_from_config:
                    def _make_safe_from_config(orig_fn):
                        def _safe_from_config(cls, config):
                            if isinstance(config, dict):
                                config.pop("quantization_config", None)
                            return orig_fn(config)
                        return _safe_from_config
                    layer_cls.from_config = classmethod(_make_safe_from_config(orig_from_config))
    except Exception:
        pass

_patch_keras_deserialization()


class ModelWrapper:
    """
    Standardized wrapper around Keras, TensorFlow, or TFLite models
    to provide a uniform predict() interface.
    """
    def __init__(self, raw_model: Any, model_type: str, model_path: str):
        self.raw_model = raw_model
        self.model_type = model_type  # 'keras', 'h5', 'savedmodel', 'tflite'
        self.model_path = model_path
        self._input_shape = None
        self._output_shape = None
        self._inspect_shapes()

    def _inspect_shapes(self):
        try:
            if self.model_type in ["keras", "h5", "savedmodel"]:
                self._input_shape = getattr(self.raw_model, "input_shape", None)
                self._output_shape = getattr(self.raw_model, "output_shape", None)
            elif self.model_type == "tflite":
                input_details = self.raw_model.get_input_details()
                output_details = self.raw_model.get_output_details()
                self._input_shape = tuple(input_details[0]["shape"])
                self._output_shape = tuple(output_details[0]["shape"])
        except Exception:
            pass

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        return self._output_shape

    def predict(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Runs inference and returns softmax prediction vector of shape (batch, num_classes).
        """
        if self.model_type in ["keras", "h5", "savedmodel"]:
            preds = self.raw_model.predict(input_tensor, verbose=0)
            return np.array(preds)
        elif self.model_type == "tflite":
            interpreter = self.raw_model
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            interpreter.set_tensor(input_details[0]["index"], input_tensor.astype(np.float32))
            interpreter.invoke()
            preds = interpreter.get_tensor(output_details[0]["index"])
            return np.array(preds)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def get_summary(self) -> Dict[str, Any]:
        """Returns structured metadata about the loaded model."""
        total_params = 0
        trainable_params = 0
        try:
            if self.model_type in ["keras", "h5"]:
                total_params = int(getattr(self.raw_model, "count_params", lambda: 0)())
        except Exception:
            pass

        file_size_mb = 0.0
        if os.path.exists(self.model_path):
            file_size_mb = round(os.path.getsize(self.model_path) / (1024 * 1024), 2)

        return {
            "model_path": self.model_path,
            "filename": os.path.basename(self.model_path),
            "model_type": self.model_type.upper(),
            "input_shape": str(self._input_shape),
            "output_shape": str(self._output_shape),
            "total_parameters": f"{total_params:,}" if total_params > 0 else "N/A",
            "file_size_mb": f"{file_size_mb} MB"
        }


def _load_model_internal(model_path: str) -> ModelWrapper:
    """Internal loading routine with format auto-detection."""
    _patch_keras_deserialization()
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    ext = p.suffix.lower()

    if ext in [".keras", ".h5"]:
        import keras
        # Load without compiling for faster inference startup
        raw = keras.models.load_model(str(p), compile=False)
        return ModelWrapper(raw, "keras" if ext == ".keras" else "h5", str(p))

    elif ext == ".tflite":
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=str(p))
        interpreter.allocate_tensors()
        return ModelWrapper(interpreter, "tflite", str(p))

    elif p.is_dir():
        # TensorFlow SavedModel directory
        import tensorflow as tf
        raw = tf.saved_model.load(str(p))
        return ModelWrapper(raw, "savedmodel", str(p))

    else:
        raise ValueError(f"Unsupported model extension: '{ext}'. Supported: .keras, .h5, .tflite, SavedModel dir.")


# Cache model in Streamlit session memory
@st.cache_resource(show_spinner=False)
def get_cached_model(model_path: str) -> Tuple[Optional[ModelWrapper], Optional[str]]:
    """
    Cached accessor for models. Returns (ModelWrapper, None) on success,
    or (None, error_message) on failure.
    """
    try:
        model = _load_model_internal(model_path)
        return model, None
    except Exception as e:
        return None, str(e)


@st.cache_resource(show_spinner=False)
def load_class_names(class_file_path: str) -> Tuple[Optional[List[str]], Optional[str]]:
    """
    Loads class names from JSON file. Returns (class_list, None) or (None, error_msg).
    """
    p = Path(class_file_path)
    if not p.exists():
        return None, f"Class names file not found at: {class_file_path}"

    try:
        with open(p, "r", encoding="utf-8") as f:
            classes = json.load(f)
        if not isinstance(classes, list) or len(classes) == 0:
            return None, "Class names file must contain a non-empty JSON list of class strings."
        return classes, None
    except Exception as e:
        return None, f"Error parsing class names JSON: {e}"


def get_available_models(models_dir: str = "models") -> Dict[str, str]:
    """
    Scans the models directory and returns a dictionary of available model options
    without displaying accuracy in the model selection label.
    """
    p = Path(models_dir)
    available = {}
    if p.exists() and p.is_dir():
        for file in p.glob("*"):
            if file.suffix.lower() in [".keras", ".h5", ".tflite"]:
                stem = file.stem.lower()
                if "baseline" in stem:
                    label = "MobileNetV2 (Baseline)"
                elif "extended" in stem:
                    label = "MobileNetV2 (PSO-Optimized Extended)"
                elif "optimized" in stem or "pso" in stem:
                    label = "MobileNetV2 (PSO-Optimized)"
                else:
                    label = f"MobileNetV2 ({file.stem.replace('_', ' ').title()})"
                available[label] = str(file)

    # Fallback to default if empty
    if not available:
        default_p = Path("baseline_mobilenetv2.keras")
        if default_p.exists():
            available["MobileNetV2 (Baseline)"] = str(default_p)

    return available
