# 🚀 CropCare AI - Complete Deployment Guide

This guide details how to deploy the **CropCare AI – Intelligent Crop Disease Detection System** to the cloud or across your local network.

---

## 📋 Pre-Flight Deployment Checklist

Your repository is pre-configured and deployment-ready:
- ✅ **Optimized Requirements (`requirements.txt`):** Includes `opencv-python-headless` for headless cloud Linux environments without GUI library crashes.
- ✅ **System Package Manifest (`packages.txt`):** Includes `libgl1` and `libglib2.0-0` for Debian/Ubuntu-based cloud platforms.
- ✅ **Repository Git Protection (`.gitignore`):** Excludes the 426 MB raw dataset (`minor_dataset/`) and temporary files while preserving the trained deep learning models (`models/`).
- ✅ **Lightweight Demo Samples (`demo_samples/`):** Contains 1 sample image per class (~0.25 MB total) so the *Choose from Test Samples* feature works out of the box online for live grading and presentations.
- ✅ **Container Architecture (`Dockerfile` & `.dockerignore`):** Multi-platform Docker configuration for containerized deployment.

---

## 🌟 Option 1: Streamlit Community Cloud (Recommended & 100% Free)

Streamlit Community Cloud is the easiest, fastest, and official way to host Streamlit applications.

### Step 1: Initialize Git and Push to GitHub

Open a terminal (PowerShell or Bash) in your project root:

```bash
# 1. Initialize git repository
git init

# 2. Add all files (the pre-configured .gitignore automatically ignores large datasets)
git add .

# 3. Commit the changes
git commit -m "Initial commit for CropCare AI cloud deployment"

# 4. Set main branch
git branch -M main

# 5. Link to your GitHub repository (replace with your repo URL)
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>.git

# 6. Push to GitHub
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io/)** and sign in with your GitHub account.
2. Click **"New app"**.
3. Select your repository: `<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>`.
4. Branch: `main`.
5. Main file path: `app.py`.
6. Click **"Deploy!"**.

Within 2 to 3 minutes, your dashboard will be live at:
`https://<YOUR_APP_NAME>.streamlit.app`

---

## 🤗 Option 2: Hugging Face Spaces (Free 16 GB CPU RAM)

Hugging Face Spaces provides a free 16 GB RAM environment, which offers great performance for TensorFlow/Keras inference.

1. Go to **[huggingface.co/spaces](https://huggingface.co/spaces)** and sign in.
2. Click **"Create new Space"**.
3. Fill in:
   - **Space name:** `cropcare-ai`
   - **License:** `apache-2.0` (or `mit`)
   - **Space SDK:** Choose **Streamlit**
   - **Space hardware:** Free (CPU basic · 2 vCPU · 16 GB RAM)
4. Clone the space to your machine:
   ```bash
   git clone https://huggingface.co/spaces/<YOUR_USERNAME>/cropcare-ai
   ```
5. Copy project files into the cloned folder (excluding `minor_dataset/`), commit, and push:
   ```bash
   git add .
   git commit -m "Deploy CropCare AI to Hugging Face Spaces"
   git push
   ```
6. Hugging Face will automatically build and launch your dashboard.

---

## 🐳 Option 3: Docker Deployment (Render, Railway, AWS, or Local)

The project includes a ready-to-build `Dockerfile`.

### Local Docker Build & Run:

```bash
# 1. Build the Docker image
docker build -t cropcare-ai .

# 2. Run container on port 8501
docker run -p 8501:8501 cropcare-ai
```
Visit `http://localhost:8501` in your browser.

### Deploying to Render.com:
1. Create a free account on **[render.com](https://render.com/)**.
2. Click **"New +"** -> **"Web Service"**.
3. Connect your GitHub repository.
4. Select **Docker** environment.
5. Click **"Create Web Service"**.

---

## 📡 Option 4: Local Network Wi-Fi Hosting (Recommended for Live ESP32-CAM Demos)

When presenting in a lab, college hall, or viva demonstration with live ESP32-CAM hardware:

### Why Local Network Hosting?
Your ESP32-CAM is assigned a private local IP (e.g., `192.168.1.50`). A public cloud server (like Streamlit Cloud) cannot access private home Wi-Fi IPs across network firewalls without complex tunneling. 

Running the dashboard locally allows **any device on your Wi-Fi (phones, tablets, laptops, ESP32-CAM)** to communicate with zero latency:

```bash
streamlit run app.py --server.address=0.0.0.0 --server.port=8501
```

### Accessing the Dashboard:
- **On host computer:** `http://localhost:8501`
- **On any phone or laptop connected to the same Wi-Fi:** `http://<YOUR_COMPUTER_IP>:8501`
  *(Find your IP by running `ipconfig` on Windows; look for IPv4 Address e.g. `192.168.1.105`)*.
- The live ESP32-CAM video feed and frame capture will work directly with zero configuration.

---

## 🌐 Connecting ESP32-CAM (Both Live Stream & Still Capture) to Streamlit Cloud via ngrok

The ESP32-CAM uses two separate ports:
- **Port 80:** HTTP Still Capture (`/capture`)
- **Port 81:** MJPEG Live Stream (`/stream`)

Because free ngrok accounts are limited to **1 simultaneous tunnel**, the project includes an automated bridge (`esp32_bridge.py`) that unites both ports onto a single endpoint.

### Method A: Single Free ngrok Tunnel (Recommended)

1. **Start the Dual-Port Bridge on your PC:**
   Open a terminal in your project directory and run (replace with your ESP32's local IP):
   ```bash
   python esp32_bridge.py 192.168.1.50
   ```
   *This unites Port 80 and Port 81 onto `http://localhost:8080`.*

2. **Start ngrok:**
   In a second terminal window, run:
   ```bash
   ngrok http 8080
   ```
   Copy the generated public URL (e.g., `https://a1b2-c3d4.ngrok-free.app`).

3. **Paste into the Cloud Dashboard:**
   - Open your deployed Streamlit Cloud app.
   - Go to **Disease Detection** -> **📷 ESP32-CAM IoT Hardware**.
   - Paste the URL `https://a1b2-c3d4.ngrok-free.app` into **ESP32-CAM IP or Public Tunnel URL**.
   - **Both the live video feed AND frame capture will work seamlessly over the internet!**

---

### Method B: Native ngrok Multi-Tunnel (`ngrok.yml`)

If your ngrok plan supports multiple tunnels, configure `ngrok.yml`:

```yaml
version: "2"
authtoken: <YOUR_AUTHTOKEN>
tunnels:
  esp-capture:
    proto: http
    addr: 192.168.1.50:80
  esp-stream:
    proto: http
    addr: 192.168.1.50:81
```

Run:
```bash
ngrok start --all
```
In your dashboard:
- Paste the `esp-capture` URL into **ESP32-CAM IP or Public Tunnel URL**.
- Paste the `esp-stream` URL into **Dedicated Stream URL (Optional)**.

---

## 🔒 Cloud vs ESP32-CAM Architecture Summary

| Feature | Cloud Deployment (Streamlit Cloud / HF) | Local Network Deployment (`--server.address=0.0.0.0`) |
| :--- | :--- | :--- |
| **Manual Image Upload** | ✅ Works everywhere globally | ✅ Works locally |
| **Test Sample Picker** | ✅ Works (`demo_samples/`) | ✅ Works (`minor_dataset/` + demo) |
| **MobileNetV2 AI Diagnosis**| ✅ Runs on cloud CPU | ✅ Runs on host computer |
| **Detection History / SQLite**| ✅ Active in cloud session | ✅ Persisted in `cropcare.db` |
| **Plotly Analytics** | ✅ Fully interactive | ✅ Fully interactive |
| **ESP32-CAM Live Video Stream**| ⚠️ Requires ngrok/port-forwarding | ✅ **Direct native Wi-Fi connection** |
