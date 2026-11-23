"""
CropCare AI - Disease Detection Page
The core diagnostic interface supporting leaf photo uploads, sample testing,
ESP32-CAM IoT acquisition, MobileNetV2 inference, and database recording.
"""

from pathlib import Path
from datetime import datetime
import os
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

from config import (
    DEFAULT_MODEL_FILE,
    CLASS_NAMES_FILE,
    SUPPORTED_IMAGE_TYPES,
    TARGET_SIZE,
    DEFAULT_PREPROCESSING,
    UPLOADS_DIR,
    DATASET_DIR,
    DEMO_SAMPLES_DIR
)
from modules.preprocessing import (
    load_and_validate_image,
    preprocess_for_inference,
    extract_image_metadata,
    validate_leaf_foliage
)
from modules.model_loader import get_cached_model, load_class_names, get_available_models
from modules.prediction import predict_disease
from modules.database import save_detection
from modules.camera import ESP32CamClient, get_hardware_specs


def render_disease_detection_page():
    """Renders the Crop Disease Detection UI."""

    st.markdown("""
        <div class="section-header">
            <span>🔬</span>
            <h3>Crop Leaf Disease Diagnosis</h3>
        </div>
        <p style="color: #6c757d; font-size: 0.9rem; margin-top: -0.5rem;">
            Upload a clear photograph of a crop leaf or capture via ESP32-CAM IoT hardware to classify pathological symptoms.
        </p>
    """, unsafe_allow_html=True)

    # 1. MODEL SELECTION & STATUS CHECK
    available_models = get_available_models()
    
    col_model_select, col_prep_select = st.columns([2, 1])
    with col_model_select:
        model_options = list(available_models.keys())
        selected_model_label = st.selectbox(
            "Select Inference Model Architecture:",
            options=model_options if model_options else ["Default MobileNetV2"],
            index=0,
            help="Choose between Baseline MobileNetV2 and PSO-Optimized MobileNetV2."
        )
        chosen_model_path = available_models.get(selected_model_label, DEFAULT_MODEL_FILE)

    with col_prep_select:
        prep_method = st.selectbox(
            "Preprocessing Scheme:",
            options=["mobilenet_v2", "rescale_255", "raw"],
            index=0,
            format_func=lambda x: "MobileNetV2 [-1, 1]" if x == "mobilenet_v2" else ("Rescale [0, 1]" if x == "rescale_255" else "Raw Pixels"),
            help="Standard MobileNetV2 uses [-1, 1] normalization matching training weights."
        )

    # Load Model and Class Labels
    model_obj, model_err = get_cached_model(chosen_model_path)
    class_names, class_err = load_class_names(CLASS_NAMES_FILE)

    # Error handling for missing model or class list
    if model_err or model_obj is None:
        st.error(f"""
            **⚠️ AI Model Missing or Inaccessible**
            
            Could not load deep learning model from `{chosen_model_path}`.
            
            **Resolution Instructions:**
            1. Place your trained `.keras` or `.h5` model file in the `models/` directory.
            2. Configure the path in `config.py` or set the `CROPCARE_MODEL_PATH` environment variable.
            3. Error details: `{model_err}`
        """)
        st.info("Disease detection is disabled until a valid trained model is loaded.")
        return

    if class_err or not class_names:
        st.error(f"""
            **⚠️ Class Labels Configuration Missing**
            
            Could not load class names from `{CLASS_NAMES_FILE}`.
            
            **Resolution:** Ensure `models/class_names.json` exists and contains the 17 class labels.
            Error: `{class_err}`
        """)
        return

    st.markdown("---")

    # 2. IMAGE SOURCE SELECTION TABS
    tab_upload, tab_sample, tab_esp32 = st.tabs([
        "📤 Manual File Upload",
        "📂 Choose from Test Samples",
        "📷 ESP32-CAM IoT Hardware"
    ])

    leaf_image = None
    image_source_name = "Upload"

    # TAB 1: FILE UPLOADER
    with tab_upload:
        st.markdown("##### Upload Crop Leaf Image")
        uploaded_file = st.file_uploader(
            "Select leaf image (supported formats: JPG, JPEG, PNG, WEBP):",
            type=SUPPORTED_IMAGE_TYPES,
            key="leaf_file_uploader",
            help="Take a clear, well-lit photo of an individual leaf showing healthy or symptomatic areas."
        )
        if uploaded_file is not None:
            try:
                leaf_image = load_and_validate_image(uploaded_file.getvalue())
                image_source_name = uploaded_file.name
            except Exception as e:
                st.error(f"Error reading image: {e}")

    # TAB 2: SAMPLE DATASET TESTER (Great for presentations!)
    with tab_sample:
        st.markdown("##### Select Demonstration Image from Dataset")
        st.caption("Convenient for offline testing and viva presentations when no custom leaf photo is at hand.")
        
        sample_options = {}
        sample_source_dir = DATASET_DIR if (DATASET_DIR.exists() and any(DATASET_DIR.iterdir())) else DEMO_SAMPLES_DIR
        if sample_source_dir.exists() and sample_source_dir.is_dir():
            for class_folder in sorted(sample_source_dir.iterdir()):
                if class_folder.is_dir():
                    images = list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.JPG")) + list(class_folder.glob("*.png"))
                    if images:
                        # Grab first image as sample
                        sample_options[class_folder.name] = str(images[0])

        if sample_options:
            selected_sample_class = st.selectbox(
                "Select Ground-Truth Class to Test:",
                options=list(sample_options.keys()),
                index=0
            )
            sample_path = sample_options[selected_sample_class]
            if st.button("Load Selected Sample Leaf"):
                try:
                    leaf_image = Image.open(sample_path).convert("RGB")
                    image_source_name = f"Sample: {selected_sample_class}"
                    st.session_state["active_sample_image"] = leaf_image
                    st.session_state["active_sample_name"] = image_source_name
                except Exception as e:
                    st.error(f"Failed to load sample image: {e}")

            if "active_sample_image" in st.session_state and uploaded_file is None:
                leaf_image = st.session_state["active_sample_image"]
                image_source_name = st.session_state.get("active_sample_name", "Dataset Sample")
        else:
            st.info("No sample images found. Use the manual upload tab.")

    # TAB 3: ESP32-CAM IOT INTEGRATION
    with tab_esp32:
        st.markdown("##### 📷 ESP32-CAM IoT Wireless Acquisition")
        st.caption("Inspect live camera video feed from your ESP32-CAM edge node over local Wi-Fi and capture leaf frames for diagnosis.")

        # Camera Network Settings Controls
        with st.expander("⚙️ Camera Connection & Streaming Configuration", expanded=False):
            cam_c1, cam_c2, cam_c3, cam_c4 = st.columns([2, 1, 1, 1.2])
            with cam_c1:
                cam_ip = st.text_input(
                    "ESP32-CAM IP or Public Tunnel URL:",
                    value=st.session_state.get("esp32_cam_ip_val", "192.168.1.50"),
                    key="esp32_cam_ip_input",
                    help="Local IP (e.g. 192.168.1.50) when running locally, or public tunnel URL (e.g. https://xxx.ngrok-free.app) for cloud deployment."
                )
                st.session_state["esp32_cam_ip_val"] = cam_ip
            with cam_c2:
                cam_port = st.number_input(
                    "HTTP Port:",
                    value=int(st.session_state.get("esp32_cam_port_val", 80)),
                    min_value=1,
                    max_value=65535,
                    key="esp32_cam_port_input",
                    help="Web server port (standard CameraWebServer uses port 80)."
                )
                st.session_state["esp32_cam_port_val"] = cam_port
            with cam_c3:
                stream_port = st.number_input(
                    "Stream Port:",
                    value=int(st.session_state.get("esp32_cam_stream_port_val", 81)),
                    min_value=1,
                    max_value=65535,
                    key="esp32_cam_stream_port_input",
                    help="MJPEG stream port (CameraWebServer defaults to port 81; single-port sketches use 80)."
                )
                st.session_state["esp32_cam_stream_port_val"] = stream_port
            with cam_c4:
                cam_path = st.text_input(
                    "Capture Path:",
                    value=st.session_state.get("esp32_cam_path_val", "/capture"),
                    key="esp32_cam_path_input",
                    help="Still snapshot path (e.g. /capture, /jpg, /saved-photo, /photo.jpg)."
                )
                st.session_state["esp32_cam_path_val"] = cam_path

            custom_stream_url = st.text_input(
                "Dedicated Stream URL (Optional):",
                value=st.session_state.get("esp32_cam_custom_stream_val", ""),
                key="esp32_cam_custom_stream_input",
                help="Leave empty if using the single-tunnel bridge or same host. If you have a separate stream URL (e.g. from ngrok), paste it here."
            )
            st.session_state["esp32_cam_custom_stream_val"] = custom_stream_url

        # If the expander is collapsed, retrieve values from session state
        cam_ip = st.session_state.get("esp32_cam_ip_val", "192.168.1.50")
        cam_port = int(st.session_state.get("esp32_cam_port_val", 80))
        stream_port = int(st.session_state.get("esp32_cam_stream_port_val", 81))
        cam_path = st.session_state.get("esp32_cam_path_val", "/capture")
        custom_stream_url = st.session_state.get("esp32_cam_custom_stream_val", "")

        cam_client = ESP32CamClient(
            ip_address=cam_ip,
            port=cam_port,
            capture_path=cam_path,
            stream_port=stream_port,
            stream_path="/stream",
            custom_stream_url=custom_stream_url if custom_stream_url.strip() else None
        )

        # LIVE VIDEO FEED VIEWPORT
        st.markdown("<br>", unsafe_allow_html=True)
        show_live_video = st.toggle("🔴 Display Live Camera Video Feed", value=True, key="esp32_live_toggle")

        if show_live_video:
            stream_url = cam_client.stream_url
            raw_addr = cam_ip.strip()
            is_local_ip = (
                raw_addr.startswith("192.168.")
                or raw_addr.startswith("10.")
                or raw_addr.startswith("172.")
                or "localhost" in raw_addr
                or "127.0.0.1" in raw_addr
            )

            # Warning if user might be on cloud HTTPS but using private LAN IP
            if is_local_ip:
                st.caption(
                    "💡 **Local Wi-Fi Mode**: If accessing this dashboard locally (`localhost:8501`), the stream connects directly. "
                    "If viewing on **Streamlit Cloud** (`https://*.streamlit.app`), browsers block private IPs (`192.168.x.x`) due to Mixed-Content security. "
                    "To view on cloud: run `python esp32_bridge.py` & `ngrok http 8080` on your PC, then enter the `https://*.ngrok-free.app` URL above."
                )

            st.markdown(f"""
                <div style="background: #0b0f19; border: 2px solid #2d6a4f; border-radius: 14px; overflow: hidden; position: relative; margin: 0.5rem 0 1rem 0; box-shadow: 0 8px 24px rgba(0,0,0,0.18);">
                    <div style="display: flex; justify-content: space-between; align-items: center; background: #131d2e; padding: 0.55rem 1.1rem; border-bottom: 1px solid #1e293b; flex-wrap: wrap; gap: 0.5rem;">
                        <div style="display: flex; align-items: center; gap: 0.6rem;">
                            <span style="display: inline-block; width: 10px; height: 10px; background: #ef4444; border-radius: 50%; box-shadow: 0 0 8px #ef4444;"></span>
                            <span style="font-weight: 800; font-size: 0.85rem; color: #ffffff; letter-spacing: 0.05em;">LIVE ESP32-CAM STREAM</span>
                        </div>
                        <div style="font-family: monospace; font-size: 0.78rem; color: #94a3b8;">
                            <a href="{stream_url}" target="_blank" style="color: #38bdf8; text-decoration: underline; margin-right: 0.8rem;">Open Stream in New Tab ↗</a>
                            <span>{stream_url}</span>
                        </div>
                    </div>
                    <div style="text-align: center; padding: 0.8rem; background: #020617; min-height: 260px; display: flex; align-items: center; justify-content: center; flex-direction: column;">
                        <img src="{stream_url}" 
                             id="esp32_live_stream_img"
                             alt="ESP32-CAM Live Feed (Connecting to video stream...)" 
                             style="max-width: 100%; max-height: 420px; border-radius: 8px; box-shadow: 0 4px 14px rgba(0,0,0,0.4); object-fit: contain;" 
                             onerror="this.style.display='none'; document.getElementById('esp32_stream_fallback').style.display='block';"
                             onload="this.style.display='block'; document.getElementById('esp32_stream_fallback').style.display='none';" />
                        <div id="esp32_stream_fallback" style="display: none; padding: 2rem 1rem; color: #94a3b8; font-size: 0.9rem;">
                            ⚠️ <b>Live Stream Waiting for ESP32-CAM Signal</b><br>
                            <span style="font-size: 0.82rem; color: #64748b; line-height: 1.6;">
                                Could not render live MJPEG stream from <code>{stream_url}</code> directly in this browser viewport.<br>
                                <b>Quick Fixes:</b><br>
                                1. Verify your ESP32-CAM is powered with a stable 5V 2A supply and connected to Wi-Fi.<br>
                                2. Click <a href="{stream_url}" target="_blank" style="color: #38bdf8;"><b>Open Stream in New Tab ↗</b></a> to test if your browser can reach the video feed.<br>
                                3. Click <b>🔴 Test Live Stream</b> below to diagnose connection response.<br>
                                4. Click <b>🔄 Fetch Live Frame</b> below to pull the current camera frame instantly via backend server!
                            </span>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            st.caption("💡 Point the ESP32-CAM camera lens steadily at the crop leaf in the live viewport above, then click **Capture Frame for Diagnosis** below.")

        # ACTION BUTTONS ROW
        btn_c1, btn_c2, btn_c3, btn_c4 = st.columns([1.6, 1.2, 1.2, 1.2])

        with btn_c1:
            if st.button("📸 Capture Frame for Diagnosis", use_container_width=True, type="primary"):
                with st.spinner("Capturing high-resolution frame from ESP32-CAM..."):
                    cam_img, cam_err = cam_client.capture_snapshot()
                    if cam_img:
                        st.success(f"Snapshot acquired successfully from ESP32-CAM ({cam_client.capture_path})!")
                        leaf_image = cam_img
                        image_source_name = f"ESP32-CAM ({cam_ip}:{cam_port}{cam_client.capture_path})"
                        st.session_state["active_sample_image"] = leaf_image
                        st.session_state["active_sample_name"] = image_source_name
                        st.rerun()
                    else:
                        st.error(f"Capture failed: {cam_err}")

        with btn_c2:
            if st.button("🔄 Fetch Live Frame", use_container_width=True, help="Pulls the latest single frame from the live video stream or snapshot"):
                with st.spinner("Fetching frame from live stream..."):
                    frame_img, frame_err = cam_client.capture_stream_frame()
                    if frame_img is None:
                        # Fallback to snapshot
                        frame_img, frame_err = cam_client.capture_snapshot()
                    if frame_img:
                        st.success("Current live frame fetched successfully!")
                        leaf_image = frame_img
                        image_source_name = f"ESP32-CAM Live Frame ({cam_ip})"
                        st.session_state["active_sample_image"] = leaf_image
                        st.session_state["active_sample_name"] = image_source_name
                        st.rerun()
                    else:
                        st.error(f"Could not fetch frame: {frame_err}")

        with btn_c3:
            if st.button("📡 Test Snapshot & Stream", use_container_width=True):
                with st.spinner("Diagnosing ESP32-CAM ports..."):
                    snap_ok, snap_msg = cam_client.test_connection()
                    stream_ok, stream_msg = cam_client.test_stream_connection()
                    
                    if snap_ok:
                        st.success(f"**Snapshot Port**: ✅ {snap_msg}")
                    else:
                        st.warning(f"**Snapshot Port**: ⚠️ {snap_msg}")
                        
                    if stream_ok:
                        st.success(f"**Live Stream Port**: ✅ {stream_msg}")
                    else:
                        st.error(f"**Live Stream Port**: ❌ {stream_msg}")

        with btn_c4:
            if st.button("🔍 Auto-Detect", use_container_width=True, help="Scan common ESP32-CAM camera endpoints"):
                with st.spinner("Scanning common ESP32-CAM endpoints..."):
                    endpoints = cam_client.probe_endpoints()
                    found = [ep for ep in endpoints if ep.get("has_jpeg")]
                    if found:
                        st.success(f"Found active image endpoint: `{found[0]['endpoint']}`")
                        st.session_state["esp32_cam_path_val"] = found[0]["endpoint"]
                        st.rerun()
                    else:
                        st.info("No standard JPEG endpoint responded automatically. Check device IP.")

        with st.expander("🛠️ ESP32-CAM Hardware, Power & Stream Troubleshooting Guide"):
            st.markdown("""
                - **Live Video Not Showing:**
                  1. Check if your sketch streams on **port 81** (default for Arduino `CameraWebServer`) or **port 80** (single-port sketches). Adjust the **Stream Port** above accordingly.
                  2. Make sure your browser and the ESP32-CAM are on the **same Wi-Fi network and subnet**.
                  3. If Chrome blocks insecure sub-resources, click the tune icon next to `localhost:8501` in your browser address bar and enable 'Insecure content' for local network streaming.
                - **'Camera capture failed' Error:**
                  1. **Power Supply:** The ESP32-CAM requires up to 500mA peak current when transmitting Wi-Fi packets. A standard 9V alkaline battery often suffers severe voltage drop (brownout) under transmission, causing the camera sensor to fail. Power the board with a stable **5V 2A USB power supply** or dedicated power adapter.
                  2. **Camera Ribbon Cable:** Ensure the OV2640 camera ribbon is firmly seated and locked in its socket.
                  3. **PSRAM Configuration:** In Arduino IDE, verify that **Tools > PSRAM** is set to **Enabled**.
                - **'Cannot reach ESP32-CAM' Error:**
                  Ensure your computer and the ESP32-CAM are connected to the same Wi-Fi subnet and SSID. Check your Arduino Serial Monitor at 115200 baud to confirm the assigned IP.
            """)

    # 3. IMAGE PREVIEW & INFERENCE PIPELINE
    if leaf_image is not None:
        # Detect if current image source has changed to reset previous diagnostic state
        current_img_id = f"{image_source_name}_{leaf_image.size}_{leaf_image.mode}"
        if st.session_state.get("active_img_source_id") != current_img_id:
            st.session_state["active_img_source_id"] = current_img_id
            st.session_state.pop("latest_result", None)
            st.session_state.pop("latest_record_id", None)
            st.session_state.pop("validation_warning", None)

        st.markdown("---")
        preview_col, action_col = st.columns([1, 1])

        with preview_col:
            st.markdown("##### 🖼️ Leaf Image Preview")
            st.image(leaf_image, caption=f"Source: {image_source_name}", use_container_width=True)
            meta = extract_image_metadata(leaf_image)
            st.caption(f"Dimensions: {meta['width']}×{meta['height']} px | Mode: {meta['mode']} | Aspect Ratio: {meta['aspect_ratio']}")

        with action_col:
            st.markdown("##### ⚡ Diagnostic Control")
            st.markdown(f"""
                - **Target Model:** `{selected_model_label}`
                - **Preprocess Dimension:** `224×224 px`
                - **Normalization:** `{prep_method}`
            """)

            run_detection = st.button("🔍 Detect Disease", type="primary", use_container_width=True)

            if st.button("🧹 Clear & Reset", use_container_width=True):
                for k in [
                    "active_sample_image",
                    "active_sample_name",
                    "latest_result",
                    "latest_record_id",
                    "validation_warning",
                    "active_img_source_id"
                ]:
                    st.session_state.pop(k, None)
                st.rerun()

        # 4. RUN INFERENCE & DISPLAY RESULTS
        if run_detection:
            # Clear previous outcomes
            st.session_state.pop("latest_result", None)
            st.session_state.pop("latest_record_id", None)
            st.session_state.pop("validation_warning", None)

            with st.spinner("Analyzing image composition and foliar characteristics..."):
                # STEP 1: Visual Foliage & Random Image Plausibility Check
                is_foliage, foliage_msg, foliar_metrics = validate_leaf_foliage(leaf_image)

                if not is_foliage:
                    st.session_state["validation_warning"] = {
                        "category": "Random / Non-Plant Image",
                        "title": "Inappropriate Image Detected",
                        "message": "The uploaded image does not appear to be a crop leaf.",
                        "detail": foliage_msg,
                        "diagnostics": foliar_metrics,
                        "advice": (
                            "Please upload a clear, focused photograph of an individual plant leaf. "
                            "Random photos, electronics, human skin, vehicles, furniture, or synthetic graphics are not supported."
                        )
                    }
                else:
                    # STEP 2: Preprocess Image & Execute MobileNetV2 Inference
                    try:
                        batch_tensor, _ = preprocess_for_inference(
                            image=leaf_image,
                            target_size=TARGET_SIZE,
                            method=prep_method
                        )

                        result = predict_disease(
                            model=model_obj,
                            input_tensor=batch_tensor,
                            class_names=class_names,
                            top_k=3
                        )

                        # STEP 3: In-Distribution 5-Crop Verification
                        if not result.get("is_valid_crop", True):
                            st.session_state["validation_warning"] = {
                                "category": "Unsupported Crop / Out-of-Distribution",
                                "title": "Unsupported Crop Detected",
                                "message": "The uploaded leaf does not belong to any of the 5 supported crops.",
                                "detail": result.get("validation_error"),
                                "diagnostics": {
                                    "model_confidence": f"{result['confidence']:.1f}%",
                                    "margin_to_next": f"{result.get('margin', 0):.1f}%",
                                    "entropy": result.get("entropy", "N/A")
                                },
                                "advice": (
                                    "This system is trained exclusively on 5 agricultural crops: "
                                    "1. Corn (Maize), 2. Grape, 3. Bell Pepper, 4. Potato, and 5. Tomato. "
                                    "Images of other plant species, weeds, or non-target crops cannot be classified."
                                )
                            }
                        else:
                            # STEP 4: Valid Crop & Pathology -> Save to disk and SQLite
                            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                            safe_filename = f"scan_{timestamp_str}.jpg"
                            saved_img_path = UPLOADS_DIR / safe_filename
                            leaf_image.save(saved_img_path, format="JPEG")

                            record_id = save_detection(
                                crop_name=result["crop_name"],
                                disease_name=result["disease_name"],
                                raw_class=result["raw_class"],
                                confidence=result["confidence"],
                                health_status=result["health_status"],
                                image_path=str(saved_img_path),
                                top_predictions=result["top_predictions"],
                                model_used=selected_model_label
                            )

                            st.session_state["latest_result"] = result
                            st.session_state["latest_record_id"] = record_id

                    except Exception as e:
                        st.error(f"Inference error: {str(e)}")
                        return

        # Render Outcome: Warning (for invalid/random/unsupported) OR Diagnostic Report (for valid)
        if "validation_warning" in st.session_state:
            warn = st.session_state["validation_warning"]
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
                <div class="section-header">
                    <span>⚠️</span>
                    <h3>Image Validation Warning</h3>
                </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
                <div style="background: #fff8f5; border: 2px solid #e76f51; border-radius: 16px; padding: 1.8rem; box-shadow: 0 4px 14px rgba(231, 111, 81, 0.12);">
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.8rem;">
                        <span style="background: #ffe3e3; color: #c92a2a; border: 1px solid #ffa8a8; padding: 0.35rem 0.9rem; border-radius: 20px; font-weight: 800; font-size: 0.85rem; letter-spacing: 0.03em;">
                            ⚠️ {warn['title'].upper()}
                        </span>
                        <span style="font-size: 0.82rem; color: #8a99a8; font-weight: 600;">{warn['category']}</span>
                    </div>
                    <h3 style="color: #9c2714; margin: 0.4rem 0 0.6rem 0; font-size: 1.35rem; font-weight: 800;">
                        {warn['message']}
                    </h3>
                    <div style="background: #ffffff; border-left: 4px solid #e76f51; padding: 0.9rem 1.1rem; border-radius: 0 10px 10px 0; margin: 1rem 0; box-shadow: 0 2px 6px rgba(0,0,0,0.03);">
                        <div style="font-size: 0.85rem; font-weight: 700; color: #264653; margin-bottom: 0.3rem;">Diagnostic Finding:</div>
                        <div style="font-size: 0.92rem; color: #495057;">{warn['detail']}</div>
                    </div>
                    <div style="background: #f4fbf6; border: 1px solid #b7e4c7; padding: 0.9rem 1.1rem; border-radius: 10px; margin-top: 0.8rem;">
                        <div style="font-weight: 700; color: #1b4332; font-size: 0.9rem; margin-bottom: 0.2rem;">💡 Actionable Guidance:</div>
                        <div style="font-size: 0.88rem; color: #2d6a4f;">{warn['advice']}</div>
                        <div style="font-size: 0.82rem; color: #52b788; margin-top: 0.4rem; font-weight: 600;">
                            Supported crops: 🌽 Corn (Maize) • 🍇 Grape • 🫑 Bell Pepper • 🥔 Potato • 🍅 Tomato
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            st.info("ℹ️ No diagnosis was produced or saved to history because the image failed validation.")

        elif "latest_result" in st.session_state:
            res = st.session_state["latest_result"]
            rec_id = st.session_state.get("latest_record_id", "N/A")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
                <div class="section-header">
                    <span>📋</span>
                    <h3>Diagnostic Report</h3>
                </div>
            """, unsafe_allow_html=True)

            is_healthy = res["is_healthy"]
            card_class = "result-card-healthy" if is_healthy else "result-card-diseased"
            status_pill_class = "status-pill-healthy" if is_healthy else "status-pill-diseased"
            status_icon = "🌿" if is_healthy else "⚠️"
            status_text = "HEALTHY FOLIAGE" if is_healthy else "PATHOLOGY DETECTED"

            # Primary Result Card
            st.markdown(f"""
                <div class="{card_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                        <span class="status-pill {status_pill_class}">
                            <span>{status_icon}</span>
                            <span>{status_text}</span>
                        </span>
                        <span style="font-size: 0.82rem; color: #6c757d; font-weight: 600;">
                            Detection Record #{rec_id}
                        </span>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1.2rem;">
                        <div>
                            <div style="font-size: 0.8rem; text-transform: uppercase; color: #6c757d; font-weight: 700;">Crop Name</div>
                            <div style="font-size: 1.6rem; font-weight: 800; color: #1b4332;">{res['crop_name']}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.8rem; text-transform: uppercase; color: #6c757d; font-weight: 700;">Condition / Disease</div>
                            <div style="font-size: 1.6rem; font-weight: 800; color: {'#2d6a4f' if is_healthy else '#d90429'};">{res['disease_name']}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.8rem; text-transform: uppercase; color: #6c757d; font-weight: 700;">Prediction Confidence</div>
                            <div style="font-size: 1.6rem; font-weight: 800; color: #1b4332;">{res['confidence']:.2f}%</div>
                        </div>
                        <div>
                            <div style="font-size: 0.8rem; text-transform: uppercase; color: #6c757d; font-weight: 700;">Health Status</div>
                            <div style="font-size: 1.6rem; font-weight: 800; color: {'#2d6a4f' if is_healthy else '#d90429'};">{res['health_status']}</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Top 3 Candidates Breakdown
            top_col, advisory_col = st.columns([1, 1])

            with top_col:
                st.markdown("##### 🏆 Top-3 Model Candidates")
                for rank, cand in enumerate(res["top_predictions"], 1):
                    cand_healthy = cand["is_healthy"]
                    bar_color = "#52b788" if cand_healthy else "#e76f51"
                    st.markdown(f"""
                        <div style="background: #ffffff; border: 1px solid #e9ecef; border-radius: 10px; padding: 0.8rem 1rem; margin-bottom: 0.6rem;">
                            <div style="display: flex; justify-content: space-between; font-weight: 700; font-size: 0.92rem; color: #1b4332;">
                                <span>#{rank} {cand['crop']} – {cand['disease']}</span>
                                <span>{cand['confidence_percent']:.2f}%</span>
                            </div>
                            <div style="background: #e9ecef; border-radius: 8px; height: 8px; margin-top: 0.4rem; overflow: hidden;">
                                <div style="background: {bar_color}; width: {min(cand['confidence_percent'], 100):.1f}%; height: 100%;"></div>
                            </div>
                            <div style="font-size: 0.75rem; color: #8a99a8; margin-top: 0.25rem;">
                                Class key: <code>{cand['raw_class']}</code>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

            with advisory_col:
                st.markdown("##### 🌾 Agronomic Advisory & Best Practices")
                adv = res.get("advisory", {})
                st.markdown(f"""
                    <div class="advisory-box">
                        <h4>Pathology Advisory</h4>
                        <p style="font-size: 0.88rem; color: #495057; margin-bottom: 0.8rem;">
                            {adv.get('summary', 'Ensure standard cultural field management.')}
                        </p>
                        <h4 style="font-size: 0.9rem; margin-top: 0.8rem;">Recommended Field Actions:</h4>
                        <ul>
                            {"".join([f"<li>{action}</li>" for action in adv.get('actions', [])])}
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
                st.caption("ℹ️ Diagnosis saved automatically to Detection History.")
        else:
            st.markdown("""
                <div class="empty-state" style="padding: 1.6rem; margin-top: 1.2rem; background: #fbfdfc; border: 1px dashed #b7e4c7; border-radius: 12px;">
                    <div class="icon">🌿</div>
                    <h4 style="color: #1b4332; font-size: 1.1rem; margin-bottom: 0.3rem;">Leaf Image Ready for Diagnosis</h4>
                    <p style="color: #495057; font-size: 0.9rem; margin: 0;">Click the green <b>Detect Disease</b> button above to analyze foliar composition and classify crop health.</p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="empty-state">
                <div class="icon">🌿</div>
                <h4>Upload or Capture a Leaf Image</h4>
                <p>Choose an image file from your computer, test an included dataset sample, or acquire a live frame via ESP32-CAM to begin classification.</p>
            </div>
        """, unsafe_allow_html=True)
