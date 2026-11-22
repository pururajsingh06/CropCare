"""
CropCare AI - Detection History Page
Provides search, multi-criteria filtering, CSV export, record inspection,
and deletion management for past crop disease diagnoses stored in SQLite.
"""

from pathlib import Path
import os
import json
import streamlit as st
import pandas as pd
from PIL import Image

from modules.database import (
    get_all_detections,
    delete_detection,
    clear_all_detections,
    export_history_csv,
    get_dashboard_statistics
)
from config import SUPPORTED_CROPS


def render_history_page():
    """Renders the Detection History interface."""

    st.markdown("""
        <div class="section-header">
            <span>📜</span>
            <h3>Detection History & Diagnostic Logs</h3>
        </div>
        <p style="color: #6c757d; font-size: 0.9rem; margin-top: -0.5rem;">
            Inspect past crop foliage scans, search by pathology, filter by date, and export records for research documentation.
        </p>
    """, unsafe_allow_html=True)

    # 1. FILTER CONTROLS BAR
    f_col1, f_col2, f_col3, f_col4 = st.columns([2, 1.2, 1.2, 1.2])

    with f_col1:
        search_query = st.text_input(
            "Search Detections:",
            placeholder="Search crop, disease, or pathology...",
            key="history_search_input"
        )

    with f_col2:
        crop_choices = ["All Crops"] + [c["name"] for c in SUPPORTED_CROPS.values()]
        selected_crop = st.selectbox(
            "Filter by Crop:",
            options=crop_choices,
            index=0,
            key="history_crop_filter"
        )

    with f_col3:
        status_filter = st.selectbox(
            "Health Status:",
            options=["All", "Healthy", "Diseased"],
            index=0,
            key="history_status_filter"
        )

    with f_col4:
        date_filter_enabled = st.checkbox("Filter by Date", value=False)
        start_date = None
        end_date = None
        if date_filter_enabled:
            d_range = st.date_input("Date Range:", value=[], key="history_date_range")
            if isinstance(d_range, (list, tuple)) and len(d_range) == 2:
                start_date, end_date = d_range[0], d_range[1]

    # Query matching records
    records = get_all_detections(
        search=search_query if search_query.strip() else None,
        crop_filter=selected_crop if selected_crop != "All Crops" else None,
        status_filter=status_filter if status_filter != "All" else None,
        date_from=start_date,
        date_to=end_date
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. ACTIONS ROW (Count summary + CSV Export + Clear)
    act_col1, act_col2, act_col3 = st.columns([2, 1, 1])

    with act_col1:
        st.markdown(f"**Found {len(records)} matching detection record(s)**")

    with act_col2:
        csv_bytes = export_history_csv()
        st.download_button(
            label="📥 Export as CSV",
            data=csv_bytes,
            file_name="cropcare_detection_history.csv",
            mime="text/csv",
            use_container_width=True
        )

    with act_col3:
        if records:
            if st.button("🗑️ Clear All History", type="secondary", use_container_width=True):
                st.session_state["show_confirm_clear"] = True

    # Confirmation prompt for clearing history
    if st.session_state.get("show_confirm_clear", False):
        st.warning("⚠️ Are you sure you want to delete ALL detection history records? This cannot be undone.")
        cc_col1, cc_col2 = st.columns(2)
        with cc_col1:
            if st.button("Yes, Clear Everything", type="primary", use_container_width=True):
                deleted_count = clear_all_detections()
                st.session_state["show_confirm_clear"] = False
                st.success(f"Cleared {deleted_count} detection record(s).")
                st.rerun()
        with cc_col2:
            if st.button("Cancel", use_container_width=True):
                st.session_state["show_confirm_clear"] = False
                st.rerun()

    # 3. DISPLAY TABLE / RECORDS
    if not records:
        st.markdown("""
            <div class="empty-state">
                <div class="icon">🔍</div>
                <h4>No Matching Detections Found</h4>
                <p>No detection records match your current search and filter criteria, or the database is currently empty.</p>
            </div>
        """, unsafe_allow_html=True)
        return

    # Render summary table
    table_data = []
    for r in records:
        table_data.append({
            "ID": f"#{r['id']}",
            "Timestamp": r["timestamp"],
            "Crop": r["crop_name"],
            "Diagnosed Disease": r["disease_name"],
            "Confidence": f"{r['confidence']:.2f}%",
            "Health Status": r["health_status"],
            "Model Used": r.get("model_used", "MobileNetV2")
        })

    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # 4. DETAILED EXPANDABLE CARDS WITH THUMBNAIL & DELETION
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### 🔍 Record Details & Leaf Imagery")

    for r in records:
        rec_id = r["id"]
        is_healthy = r["health_status"] == "Healthy"
        status_color = "#2d6a4f" if is_healthy else "#d90429"

        with st.expander(f"Record #{rec_id}: {r['crop_name']} – {r['disease_name']} ({r['confidence']:.2f}%) • {r['timestamp']}"):
            exp_col1, exp_col2 = st.columns([1, 2])

            with exp_col1:
                img_path = r.get("image_path")
                if img_path and os.path.exists(img_path):
                    try:
                        thumb = Image.open(img_path)
                        st.image(thumb, caption=f"Scan #{rec_id}", use_container_width=True)
                    except Exception:
                        st.caption("Leaf thumbnail could not be loaded.")
                else:
                    st.caption("No image file reference saved.")

            with exp_col2:
                st.markdown(f"""
                    - **Crop:** **{r['crop_name']}**
                    - **Diagnosis:** **{r['disease_name']}**
                    - **Confidence:** `{r['confidence']:.2f}%`
                    - **Health Status:** <span style="color: {status_color}; font-weight: 700;">{r['health_status']}</span>
                    - **Dataset Class Key:** `{(r.get('raw_class') or 'N/A')}`
                    - **Model Architecture:** `{r.get('model_used', 'MobileNetV2')}`
                    - **Recorded At:** `{r['timestamp']}`
                """, unsafe_allow_html=True)

                # Show top predictions if available
                top_json = r.get("top_predictions_json")
                if top_json:
                    try:
                        top_preds = json.loads(top_json)
                        if isinstance(top_preds, list):
                            st.markdown("**Top Ranked Classes:**")
                            for c in top_preds:
                                st.write(f"- {c.get('crop')} – {c.get('disease')}: **{c.get('confidence_percent', 0):.2f}%**")
                    except Exception:
                        pass

                st.markdown("<br>", unsafe_allow_html=True)
                if st.button(f"🗑️ Delete Record #{rec_id}", key=f"del_btn_{rec_id}"):
                    if delete_detection(rec_id):
                        st.success(f"Record #{rec_id} deleted successfully.")
                        st.rerun()
                    else:
                        st.error("Failed to delete record.")
