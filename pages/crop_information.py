"""
CropCare AI - Crop Information Page
Provides comprehensive agronomic profiles and pathology guides for the
5 supported crops across the 17 trained model classes.
"""

from typing import Dict, Any
import streamlit as st
from config import SUPPORTED_CROPS, parse_class_label
from modules.model_loader import load_class_names
from config import CLASS_NAMES_FILE


# Detailed pathology knowledge base strictly matching the 17 classes
CROP_PATHOLOGY_DETAILS = {
    "Corn": {
        "diseases": [
            {
                "name": "Cercospora Leaf Spot / Gray Leaf Spot",
                "raw_class": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
                "causal_agent": "Cercospora zeae-maydis (Fungus)",
                "symptoms": "Tan to gray rectangular necrotic lesions strictly bounded by leaf veins; lesions coalesce during warm, overcast conditions.",
                "prevention": "Crop rotation with non-host crops, conventional tillage of infected debris, planting tolerant hybrids, and optimizing plant density for leaf drying."
            },
            {
                "name": "Common Rust",
                "raw_class": "Corn_(maize)___Common_rust_",
                "causal_agent": "Puccinia sorghi (Fungus)",
                "symptoms": "Prominent cinnamon-brown to reddish pustules scattered over both leaf surfaces, rupturing epidermis to release powdery urediniospores.",
                "prevention": "Utilizing resistant hybrids, early planting to avoid peak mid-summer airborne spore influx, and avoiding prolonged foliage dampness."
            },
            {
                "name": "Northern Leaf Blight",
                "raw_class": "Corn_(maize)___Northern_Leaf_Blight",
                "causal_agent": "Exserohilum turcicum (Fungus)",
                "symptoms": "Long elliptical, cigar-shaped grayish-green lesions that turn tan; olive-green fuzzy spore mats appear under prolonged dampness.",
                "prevention": "Two-year rotation with non-grass crops, deep incorporation of crop residues, and field sanitation."
            },
            {
                "name": "Healthy Corn Foliage",
                "raw_class": "Corn_(maize)___healthy",
                "causal_agent": "N/A (Normal Physiological State)",
                "symptoms": "Deep emerald green leaves with intact venation, smooth edges, and no chlorotic or necrotic spots.",
                "prevention": "Maintain adequate NPK balance, proper drainage, and scout regularly for early-stage stress."
            }
        ]
    },
    "Grape": {
        "diseases": [
            {
                "name": "Black Rot",
                "raw_class": "Grape___Black_rot",
                "causal_agent": "Guignardia bidwellii (Fungus)",
                "symptoms": "Small circular reddish-brown spots with dark margins on foliage; black pin-head pycnidia form inside spots; berries shrivel into hard black mummies.",
                "prevention": "Canopy thinning to ensure airflow, meticulous removal and burning of mummified berries during dormancy, and drip irrigation."
            },
            {
                "name": "Esca (Black Measles)",
                "raw_class": "Grape___Esca_(Black_Measles)",
                "causal_agent": "Fomitiporia mediterranea / Phaeomoniella complex",
                "symptoms": "Distinctive 'tiger-stripe' leaf chlorosis and necrosis between primary veins; dark spots on fruit skin; internal wood decay.",
                "prevention": "Pruning wound protection with tree sealants, disinfecting shears between cuts, and removing severely collapsed vines from the vineyard."
            },
            {
                "name": "Leaf Blight (Isariopsis Spot)",
                "raw_class": "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
                "causal_agent": "Pseudocercospora cladosporioides (Fungus)",
                "symptoms": "Irregular dark brown to purplish lesions with sharp demarcations; severe defoliation during rainy periods late in the growing season.",
                "prevention": "Clearing leaf litter from trellis base, open canopy training, and balanced potassium supplementation."
            },
            {
                "name": "Healthy Grape Foliage",
                "raw_class": "Grape___healthy",
                "causal_agent": "N/A (Normal Physiological State)",
                "symptoms": "Broad, vibrant palmate green leaves without interveinal mottling, scorching, or lesions.",
                "prevention": "Regular trellis maintenance, drip irrigation, and timely seasonal canopy shoots hedging."
            }
        ]
    },
    "Pepper": {
        "diseases": [
            {
                "name": "Bacterial Spot",
                "raw_class": "Pepper__bell___Bacterial_spot",
                "causal_agent": "Xanthomonas campestris pv. vesicatoria (Bacteria)",
                "symptoms": "Small, water-soaked circular lesions that turn dark brown with narrow yellow halos; premature defoliation exposing fruit to sunscald.",
                "prevention": "Use certified pathogen-free seeds, avoid overhead watering, never cultivate wet plants, and rotate away from solanaceous crops for 2-3 years."
            },
            {
                "name": "Healthy Bell Pepper Foliage",
                "raw_class": "Pepper__bell___healthy",
                "causal_agent": "N/A (Normal Physiological State)",
                "symptoms": "Uniform glossy green leaves, intact petioles, and robust vegetative structure without curling or water-soaked flecks.",
                "prevention": "Maintain moist but well-aerated loam soil, adequate calcium levels, and mulch around plant beds."
            }
        ]
    },
    "Potato": {
        "diseases": [
            {
                "name": "Early Blight",
                "raw_class": "Potato___Early_blight",
                "causal_agent": "Alternaria solani (Fungus)",
                "symptoms": "Concentric dark brown rings producing a 'target-board' pattern on older, lower foliage; surrounded by chlorotic yellow halos.",
                "prevention": "Ground-level irrigation, organic mulching to prevent soil splash onto lower leaves, and rotation with non-solanaceous crops."
            },
            {
                "name": "Late Blight",
                "raw_class": "Potato___Late_blight",
                "causal_agent": "Phytophthora infestans (Oomycete)",
                "symptoms": "Water-soaked irregular dark spots rapidly expanding into purplish-black lesions; white moldy growth visible on leaf undersides in humid mornings.",
                "prevention": "Planting certified disease-free seed tubers, prompt roguing and disposal of volunteer plants, avoiding poorly drained hollows."
            },
            {
                "name": "Healthy Potato Foliage",
                "raw_class": "Potato___healthy",
                "causal_agent": "N/A (Normal Physiological State)",
                "symptoms": "Lush, compound pinnate leaves with vibrant green coloration and smooth leaf margins free from dark spotting.",
                "prevention": "Hill up soil around stems properly, ensure steady moderate watering, and avoid excess nitrogen."
            }
        ]
    },
    "Tomato": {
        "diseases": [
            {
                "name": "Bacterial Spot",
                "raw_class": "Tomato_Bacterial_spot",
                "causal_agent": "Xanthomonas perforans / euvesicatoria (Bacteria)",
                "symptoms": "Small angular dark brown lesions surrounded by prominent yellowing halos; can cause extensive foliage shedding.",
                "prevention": "Drip irrigation, hot-water seed treatment, clean seedling trays, and strict field sanitation."
            },
            {
                "name": "Early Blight",
                "raw_class": "Tomato_Early_blight",
                "causal_agent": "Alternaria linariae (Fungus)",
                "symptoms": "Dark brown circular spots with concentric rings on lower leaves; yellowing of tissue surrounding the lesion.",
                "prevention": "Stake plants for upright growth, prune bottom leaves within 12 inches of soil, and apply mulch."
            },
            {
                "name": "Late Blight",
                "raw_class": "Tomato_Late_blight",
                "causal_agent": "Phytophthora infestans (Oomycete)",
                "symptoms": "Greasy, water-soaked dark gray/black patches that spread rapidly across leaves and stems with white sporulation on undersides.",
                "prevention": "Maximize air circulation through staking, avoid handling wet plants, destroy infected plant tissue immediately."
            },
            {
                "name": "Healthy Tomato Foliage",
                "raw_class": "Tomato_healthy",
                "causal_agent": "N/A (Normal Physiological State)",
                "symptoms": "Aromatic, deeply lobed green leaves with fine glandular hairs, sturdy petioles, and robust flower clusters.",
                "prevention": "Ensure 6–8 hours of full sunlight, consistent root-zone moisture, and balanced calcium-rich nutrition."
            }
        ]
    }
}


def render_crop_information_page():
    """Renders the Crop Information and Pathology Catalog."""

    st.markdown("""
        <div class="section-header">
            <span>📚</span>
            <h3>Crop Profiles & Disease Pathology Catalog</h3>
        </div>
        <p style="color: #6c757d; font-size: 0.9rem; margin-top: -0.5rem;">
            Explore botanical specifications and symptom identification guidelines for the 5 crops and 17 classes supported by the trained MobileNetV2 model.
        </p>
    """, unsafe_allow_html=True)

    # Crop selector tabs
    crop_keys = list(SUPPORTED_CROPS.keys())
    tabs = st.tabs([f"{SUPPORTED_CROPS[k]['icon']} {SUPPORTED_CROPS[k]['name']}" for k in crop_keys])

    for i, crop_key in enumerate(crop_keys):
        with tabs[i]:
            crop_meta = SUPPORTED_CROPS[crop_key]
            pathology_info = CROP_PATHOLOGY_DETAILS.get(crop_key, {})

            # Crop Overview Header Card
            st.markdown(f"""
                <div class="crop-card">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div>
                            <h2 style="color: #1b4332; margin: 0; font-size: 1.6rem;">
                                {crop_meta['icon']} {crop_meta['name']}
                            </h2>
                            <p style="color: #52b788; font-style: italic; margin: 0.2rem 0 0.8rem 0; font-weight: 600;">
                                {crop_meta['scientific_name']} • {crop_meta['category']}
                            </p>
                        </div>
                    </div>
                    <p style="color: #495057; font-size: 0.92rem; line-height: 1.5; margin-bottom: 0.8rem;">
                        {crop_meta['description']}
                    </p>
                    <div style="background: #f4fbf6; border-left: 3px solid #52b788; padding: 0.6rem 0.9rem; border-radius: 0 8px 8px 0; font-size: 0.85rem; color: #2d6a4f;">
                        <b>🌱 Optimal Growth Conditions:</b> {crop_meta['growth_conditions']}
                    </div>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("##### 🔍 Supported Model Classes & Pathologies")

            diseases = pathology_info.get("diseases", [])
            for d in diseases:
                is_healthy = "healthy" in d["name"].lower()
                badge_bg = "#d8f3dc" if is_healthy else "#ffe3e3"
                badge_fg = "#1b4332" if is_healthy else "#c92a2a"
                badge_border = "#74c69d" if is_healthy else "#ffa8a8"
                icon = "🌿" if is_healthy else "🦠"

                with st.expander(f"{icon} {d['name']} — [Model Class: {d['raw_class']}]", expanded=False):
                    st.markdown(f"""
                        <div style="margin-bottom: 0.8rem;">
                            <span style="background: {badge_bg}; color: {badge_fg}; border: 1px solid {badge_border}; padding: 0.2rem 0.7rem; border-radius: 12px; font-size: 0.78rem; font-weight: 700;">
                                {d['causal_agent']}
                            </span>
                        </div>
                        <p style="margin-bottom: 0.4rem;"><b>🔎 Diagnostic Symptoms:</b></p>
                        <p style="color: #495057; font-size: 0.9rem; margin-top: -0.2rem;">{d['symptoms']}</p>
                        <p style="margin-bottom: 0.4rem;"><b>🛡️ Prevention & Management:</b></p>
                        <p style="color: #495057; font-size: 0.9rem; margin-top: -0.2rem;">{d['prevention']}</p>
                    """, unsafe_allow_html=True)

    st.markdown("---")
    st.info("💡 **Model Verification Note:** All 17 classes presented in this catalog map directly to the output layer neurons of the trained MobileNetV2 architecture.")
