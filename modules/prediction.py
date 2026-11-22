"""
CropCare AI - Disease Prediction Module
Executes inference on preprocessed leaf images, extracts confidence scores,
identifies top-k candidate classes, and formats agronomic results.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from config import (
    parse_class_label,
    MIN_CONFIDENCE_THRESHOLD,
    MIN_PREDICTION_MARGIN,
    MAX_PREDICTION_ENTROPY,
    SUPPORTED_CROPS
)
from modules.model_loader import ModelWrapper


# Agronomic recommendations and advisory guide based on detected condition
AGRONOMIC_ADVISORY = {
    "Healthy Leaf": {
        "summary": "No active disease symptoms detected on the leaf tissue.",
        "actions": [
            "Maintain regular balanced irrigation; avoid overwatering or waterlogging the root zone.",
            "Continue periodic scouting every 5–7 days for early signs of pathogen arrival.",
            "Apply balanced N-P-K fertilizer according to crop growth stage guidelines.",
            "Ensure proper canopy aeration and weed management around crop beds."
        ],
        "severity": "None",
        "badge_color": "green"
    },
    "Common Rust": {
        "summary": "Fungal infection caused by Puccinia sorghi, marked by golden-brown to cinnamon pustules on upper and lower leaf surfaces.",
        "actions": [
            "Promptly rogue and destroy heavily infested lower leaves to reduce airborne spore dispersal.",
            "Avoid overhead sprinkler irrigation; wet leaf canopies encourage rapid spore germination.",
            "Ensure adequate field spacing to enhance sunlight penetration and leaf drying.",
            "Rotate with non-host crops and select rust-tolerant hybrid cultivars for upcoming planting cycles."
        ],
        "severity": "Moderate to High",
        "badge_color": "amber"
    },
    "Cercospora Leaf Spot / Gray Leaf Spot": {
        "summary": "Fungal disease caused by Cercospora zeae-maydis, producing rectangular, tan-to-gray lesions restricted by leaf veins.",
        "actions": [
            "Deep-plow crop residues after harvest to bury overwintering fungal inoculum.",
            "Implement a minimum 2-year crop rotation with non-host crops like soybeans or legumes.",
            "Improve field drainage and plant at recommended population densities to avoid stagnant humidity.",
            "Scout canopy weekly starting around the tasseling stage when susceptibility peaks."
        ],
        "severity": "High",
        "badge_color": "red"
    },
    "Northern Leaf Blight": {
        "summary": "Destructive fungal disease (Exserohilum turcicum) characterized by large, cigar-shaped, grayish-green to tan lesions.",
        "actions": [
            "Sanitize equipment and till infected stubble post-harvest to disrupt pathogen survival.",
            "Promote field ventilation by maintaining recommended row spacing.",
            "Utilize certified resistant seed hybrids adapted to your agro-climatic region.",
            "Avoid excessive late-season nitrogen applications which promote lush, vulnerable leaf growth."
        ],
        "severity": "High",
        "badge_color": "red"
    },
    "Black Rot": {
        "summary": "Devastating fungal infection (Guignardia bidwellii) causing reddish-brown circular spots with dark margins on grape leaves and mummified fruit.",
        "actions": [
            "Prune canopy vigorously to enhance airflow and accelerate morning dew evaporation.",
            "Remove and burn all mummified berries and infected canes during dormant winter pruning.",
            "Direct drip irrigation to ground level; keep foliage dry during critical shoot growth periods.",
            "Maintain clean vineyard floor by mowing or cultivating cover crops."
        ],
        "severity": "Very High",
        "badge_color": "red"
    },
    "Esca (Black Measles)": {
        "summary": "Complex fungal vascular decline disease resulting in distinctive 'tiger-stripe' interveinal chlorosis and necrotic margins.",
        "actions": [
            "Disinfect pruning shears regularly with 70% alcohol between vines to prevent spreading trunk pathogens.",
            "Seal large pruning wounds promptly with wound sealant or pruning paint during dry weather.",
            "Mark symptomatic vines during summer for careful remedial pruning during winter dormancy.",
            "Avoid heavy mechanical vine stress and water stress during peak summer heat."
        ],
        "severity": "Severe / Chronic",
        "badge_color": "red"
    },
    "Leaf Blight (Isariopsis Spot)": {
        "summary": "Fungal foliar disease causing irregular dark reddish-brown lesions with distinct dark borders, leading to premature leaf drop.",
        "actions": [
            "Rake and eliminate fallen leaf litter beneath the trellis to destroy overwintering fruiting bodies.",
            "Thin out dense leaf canopies around the fruiting zone to facilitate drying.",
            "Ensure balanced potassium and phosphorus nutrition to bolster foliage cell wall resilience.",
            "Inspect vine rows regularly following prolonged rainfall or dense foggy mornings."
        ],
        "severity": "Moderate",
        "badge_color": "amber"
    },
    "Bacterial Spot": {
        "summary": "Bacterial pathogen (Xanthomonas spp.) producing water-soaked circular lesions that turn dark brown with chlorotic yellow halos.",
        "actions": [
            "Never work or cultivate in the field while plants are wet with rain or dew to prevent physical transmission.",
            "Utilize clean, hot-water-treated certified pathogen-free seeds and disease-free transplants.",
            "Switch immediately to drip or furrow irrigation to eliminate splash dispersal of bacteria.",
            "Practice 2- to 3-year crop rotation avoiding all nightshade (Solanaceae) family members."
        ],
        "severity": "High",
        "badge_color": "red"
    },
    "Early Blight": {
        "summary": "Fungal disease caused by Alternaria solani, characterized by distinct concentric rings forming a 'target-board' pattern on older foliage.",
        "actions": [
            "Prune off the lowest 12 inches of leaves once plants are established to prevent soil-splash contamination.",
            "Apply organic mulch (straw or plastic film) around plant bases to create a physical barrier against soil-borne spores.",
            "Water only at ground level early in the morning so accidental splashes dry quickly.",
            "Rotate fields with non-solanaceous crops such as corn, brassicas, or legumes for at least two years."
        ],
        "severity": "Moderate to High",
        "badge_color": "amber"
    },
    "Late Blight": {
        "summary": "Devastating oomycete pathogen (Phytophthora infestans) that causes rapidly expanding water-soaked dark lesions with white fungal growth under humid conditions.",
        "actions": [
            "Immediately destroy and securely bag heavily infected plants to stop airborne sporangia spread.",
            "Ensure rapid soil drainage and strictly avoid water ponding around root systems.",
            "Do not compost infected foliage; destroy or deeply bury affected tissue outside cropping areas.",
            "Monitor local agricultural weather alerts for high-risk cool, damp weather advisories."
        ],
        "severity": "Critically High",
        "badge_color": "red"
    }
}


def predict_disease(
    model: ModelWrapper,
    input_tensor: np.ndarray,
    class_names: List[str],
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Executes inference and returns full structured prediction diagnostics.

    Args:
        model: ModelWrapper instance.
        input_tensor: float32 ndarray of shape (1, 224, 224, 3).
        class_names: List of raw class names (length 17).
        top_k: Number of candidate predictions to return.

    Returns:
        Dictionary containing top prediction, candidate ranks, and agronomic insights.
    """
    if model is None:
        raise ValueError("Model is not loaded. Cannot run inference.")

    if not class_names:
        raise ValueError("Class names list is empty or not loaded.")

    # Model inference
    raw_preds = model.predict(input_tensor)
    if raw_preds.ndim > 1:
        probs = raw_preds[0]
    else:
        probs = raw_preds

    probs = np.array(probs, dtype=np.float64)

    # If outputs are not yet probabilities (e.g. unnormalized logits), apply softmax
    if np.any(probs < 0) or not np.isclose(np.sum(probs), 1.0, atol=1e-2):
        exp_p = np.exp(probs - np.max(probs))
        probs = exp_p / np.sum(exp_p)

    # Sort descending
    sorted_indices = np.argsort(probs)[::-1]
    top_k = min(top_k, len(sorted_indices))

    candidates = []
    for idx in sorted_indices[:top_k]:
        raw_label = class_names[idx]
        parsed = parse_class_label(raw_label)
        p = float(probs[idx])
        candidates.append({
            "class_index": int(idx),
            "raw_class": raw_label,
            "crop": parsed["crop"],
            "disease": parsed["disease"],
            "status": parsed["status"],
            "is_healthy": parsed["is_healthy"],
            "probability": p,
            "confidence_percent": round(p * 100.0, 2)
        })

    # Top prediction details
    top = candidates[0]
    disease_key = top["disease"]

    # Calculate Out-Of-Distribution / Crop Plausibility Metrics
    top1_prob = float(probs[sorted_indices[0]])
    top2_prob = float(probs[sorted_indices[1]]) if len(sorted_indices) > 1 else 0.0
    confidence_pct = round(top1_prob * 100.0, 2)
    margin_pct = round((top1_prob - top2_prob) * 100.0, 2)
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)))

    # In-Distribution Crop Verification: Check if prediction satisfies confidence and separation thresholds
    is_valid_crop = True
    validation_error = None

    if confidence_pct < MIN_CONFIDENCE_THRESHOLD:
        is_valid_crop = False
        validation_error = (
            f"Low prediction confidence ({confidence_pct:.1f}% vs {MIN_CONFIDENCE_THRESHOLD:.0f}% minimum required). "
            "The leaf image does not reliably match any of the 5 supported crops (Corn, Grape, Bell Pepper, Potato, Tomato)."
        )
    elif margin_pct < MIN_PREDICTION_MARGIN:
        is_valid_crop = False
        validation_error = (
            f"Ambiguous prediction margin ({margin_pct:.1f}% gap between top candidate and alternative, minimum expected: {MIN_PREDICTION_MARGIN:.0f}%). "
            "The image appears to be an unsupported plant species or non-conforming leaf."
        )
    elif entropy > MAX_PREDICTION_ENTROPY:
        is_valid_crop = False
        validation_error = (
            f"High prediction entropy ({entropy:.2f} vs {MAX_PREDICTION_ENTROPY:.2f} maximum allowed), indicating dispersed probabilities across dissimilar classes. "
            "The input does not correspond to the trained 5-crop dataset."
        )

    # Retrieve matching agronomic advisory
    advisory = AGRONOMIC_ADVISORY.get(
        disease_key,
        {
            "summary": f"Detected {disease_key} on {top['crop']}.",
            "actions": [
                "Isolate affected plants to limit potential transmission.",
                "Inspect surrounding crops for similar foliar symptoms.",
                "Ensure proper air circulation and avoid excess moisture on leaves."
            ],
            "severity": "Moderate" if not top["is_healthy"] else "None",
            "badge_color": "green" if top["is_healthy"] else "amber"
        }
    )

    return {
        "crop_name": top["crop"],
        "disease_name": top["disease"],
        "confidence": top["confidence_percent"],
        "health_status": top["status"],
        "is_healthy": top["is_healthy"],
        "raw_class": top["raw_class"],
        "class_index": top["class_index"],
        "top_predictions": candidates,
        "all_probabilities": [round(float(p) * 100.0, 2) for p in probs],
        "advisory": advisory,
        "is_valid_crop": is_valid_crop,
        "validation_error": validation_error,
        "margin": margin_pct,
        "entropy": round(entropy, 3)
    }
