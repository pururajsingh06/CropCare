"""
CropCare AI - ESP32-CAM Integration Module
Provides edge-to-server image acquisition from an ESP32-CAM microcontroller module.
Handles HTTP snapshot retrieval, format conversion, and connection diagnostics.
"""

from typing import Tuple, Optional, List
import io
import urllib.request
import urllib.error
from PIL import Image
import numpy as np
import cv2
from config import (
    DEFAULT_CAMERA_IP,
    DEFAULT_CAMERA_PORT,
    CAMERA_CAPTURE_PATH,
    CAMERA_STREAM_PATH,
    CAMERA_TIMEOUT
)

# Standard endpoints used by popular ESP32-CAM Arduino/MicroPython sketches
KNOWN_CAMERA_ENDPOINTS = [
    "/capture",       # Standard Arduino CameraWebServer (OV2640 still)
    "/jpg",           # Alternate ESP32-CAM web server sketch
    "/saved-photo",   # ESP32-CAM SD/SPIFFS snapshot handler
    "/photo.jpg",     # Common static capture endpoint
    "/snapshot",      # MicroPython / general ESP32 snapshot
    "/cam-hi.jpg",    # Ai-Thinker high-res still
    "/cam.jpg",       # Generic ESP32 CAM still
    "/stream"         # MJPEG live stream (frame extraction)
]


def decode_image_payload(payload: bytes) -> Tuple[Optional[Image.Image], Optional[str]]:
    """
    Decodes raw HTTP byte payload into a valid RGB PIL Image.
    Employs JPEG SOI/EOI marker slicing (to handle multipart MJPEG boundaries and headers),
    PIL decoding, and OpenCV fallback.

    Returns:
        Tuple of (Image, error_message)
    """
    if not payload:
        return None, "Empty payload received (0 bytes)."

    lower_excerpt = payload[:500].lower()

    # 1. Check for ESP32 hardware/firmware error strings sent as HTTP 200 text
    if b"camera capture failed" in lower_excerpt or b"failed to get frame" in lower_excerpt:
        return None, (
            "ESP32-CAM Hardware Error: The board returned 'Camera capture failed'. "
            "The OV2640 sensor could not acquire a frame buffer. "
            "Remedies: 1) Ensure sufficient 5V 2A power (9V block batteries often drop voltage under Wi-Fi load), "
            "2) Check the camera ribbon cable connection to the board, "
            "3) In Arduino IDE, verify Tools > PSRAM is set to 'Enabled'."
        )

    # 2. Check if an HTML page was returned (e.g. wrong IP, router login, or index page)
    if b"<!doctype" in lower_excerpt or b"<html" in lower_excerpt or b"<body" in lower_excerpt:
        return None, (
            "The camera IP returned an HTML web page instead of JPEG image bytes. "
            "This usually means the IP address belongs to your home router or another network device, "
            "or the snapshot path is different from the web control interface. "
            "Check your Arduino Serial Monitor for the exact assigned IP address."
        )

    # 3. Locate JPEG Start-of-Image (SOI: 0xFF, 0xD8) and End-of-Image (EOI: 0xFF, 0xD9)
    soi_idx = payload.find(b"\xff\xd8")
    if soi_idx != -1:
        eoi_idx = payload.find(b"\xff\xd9", soi_idx)
        if eoi_idx != -1:
            clean_bytes = payload[soi_idx:eoi_idx + 2]
        else:
            clean_bytes = payload[soi_idx:]

        # A) Primary decode via Pillow
        try:
            img = Image.open(io.BytesIO(clean_bytes))
            # Verify and convert to RGB
            img.load()
            return img.convert("RGB"), None
        except Exception:
            pass

        # B) Secondary resilient decode via OpenCV (handles partial or non-standard headers)
        try:
            np_arr = np.frombuffer(clean_bytes, dtype=np.uint8)
            bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if bgr is not None and bgr.size > 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                return Image.fromarray(rgb), None
        except Exception:
            pass

    # 4. Check if OpenCV can decode the raw buffer anyway (PNG, WebP, BMP)
    try:
        np_arr = np.frombuffer(payload, dtype=np.uint8)
        bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if bgr is not None and bgr.size > 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb), None
    except Exception:
        pass

    # 5. Fallback direct PIL attempt
    try:
        img = Image.open(io.BytesIO(payload))
        return img.convert("RGB"), None
    except Exception as pil_err:
        snippet = payload[:80].decode("ascii", errors="replace").strip()
        return None, (
            f"Unable to decode image payload ({len(payload)} bytes). "
            f"Data begins with: {snippet!r}. Error: {pil_err}"
        )


class ESP32CamClient:
    """
    Client for acquiring live crop leaf snapshots from an ESP32-CAM board
    running standard CameraWebServer or custom snapshot server firmware.
    """

    def __init__(
        self,
        ip_address: str = DEFAULT_CAMERA_IP,
        port: int = DEFAULT_CAMERA_PORT,
        capture_path: str = CAMERA_CAPTURE_PATH,
        stream_port: Optional[int] = None,
        stream_path: str = CAMERA_STREAM_PATH,
        custom_stream_url: Optional[str] = None,
        timeout: int = CAMERA_TIMEOUT
    ):
        self.ip_address = ip_address.strip()
        self.port = port
        self.capture_path = capture_path.strip() if capture_path.strip().startswith("/") else f"/{capture_path.strip()}"
        self.custom_stream_url = custom_stream_url.strip() if custom_stream_url and custom_stream_url.strip() else None
        
        # Standard Arduino CameraWebServer runs HTTP UI on port 80 and MJPEG stream on port 81
        if stream_port is not None:
            self.stream_port = stream_port
        elif self.port == 80:
            self.stream_port = 81
        else:
            self.stream_port = self.port

        self.stream_path = stream_path.strip() if stream_path.strip().startswith("/") else f"/{stream_path.strip()}"
        self.timeout = timeout

    @property
    def base_url(self) -> str:
        addr = self.ip_address.strip()
        if addr.startswith("http://") or addr.startswith("https://"):
            return addr.rstrip("/")
        if self.port == 80:
            return f"http://{addr}"
        return f"http://{addr}:{self.port}"

    @property
    def capture_url(self) -> str:
        return f"{self.base_url}{self.capture_path}"

    @property
    def stream_url(self) -> str:
        if self.custom_stream_url:
            return self.custom_stream_url
        addr = self.ip_address.strip()
        if addr.startswith("http://") or addr.startswith("https://"):
            return f"{addr.rstrip('/')}{self.stream_path}"
        if self.stream_port == 80:
            return f"http://{addr}{self.stream_path}"
        return f"http://{addr}:{self.stream_port}{self.stream_path}"

    def capture_stream_frame(self) -> Tuple[Optional[Image.Image], Optional[str]]:
        """
        Captures a single still frame directly from the MJPEG live video stream.
        """
        try:
            req = urllib.request.Request(
                self.stream_url,
                headers={"User-Agent": "CropCare-AI-Client"}
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                # Read enough bytes from the stream to contain at least 1 complete JPEG frame (typically 30-100 KB)
                stream_buffer = response.read(256 * 1024)
                if not stream_buffer:
                    return None, "Empty stream response received from ESP32-CAM."

                img, err = decode_image_payload(stream_buffer)
                if img is not None:
                    return img, None
                return None, err or "Could not extract a valid JPEG frame from live stream."
        except urllib.error.URLError as e:
            return None, f"Could not connect to live stream at {self.stream_url}: {e.reason}"
        except Exception as e:
            return None, f"Live stream capture error: {str(e)}"

    def test_connection(self) -> Tuple[bool, str]:
        """
        Pings the ESP32-CAM and probes the capture endpoint to verify network reachability.
        Returns: (is_reachable: bool, status_message: str)
        """
        try:
            # 1. Test base URL connectivity
            req = urllib.request.Request(
                self.base_url,
                headers={"User-Agent": "CropCare-AI-Client"}
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                status_code = response.status

            if status_code != 200:
                return False, f"Camera responded with unexpected HTTP status {status_code}."

            # 2. Probe capture endpoint
            cap_req = urllib.request.Request(
                self.capture_url,
                headers={"User-Agent": "CropCare-AI-Client"}
            )
            with urllib.request.urlopen(cap_req, timeout=self.timeout) as cap_res:
                content_type = cap_res.headers.get_content_type()
                raw_head = cap_res.read(512)

            if b"camera capture failed" in raw_head.lower():
                return False, (
                    f"Connected to ESP32 at {self.base_url}, but camera sensor reported: "
                    "'Camera capture failed' (OV2640 brownout or loose ribbon cable)."
                )

            if "image" in content_type or raw_head.find(b"\xff\xd8") != -1:
                return True, f"Connected to ESP32-CAM at {self.base_url}! Endpoint '{self.capture_path}' is ready."

            return True, f"Connected to {self.base_url} (HTTP 200 OK). Endpoint '{self.capture_path}' responded with {content_type}."

        except urllib.error.HTTPError as e:
            return False, f"HTTP Error {e.code} at {self.capture_url}: {e.reason}."
        except urllib.error.URLError as e:
            raw_addr = self.ip_address.strip()
            is_private = (
                raw_addr.startswith("192.168.")
                or raw_addr.startswith("10.")
                or raw_addr.startswith("172.16.")
                or "localhost" in raw_addr
                or "127.0.0.1" in raw_addr
            )
            if "timed out" in str(e).lower() and is_private:
                return False, (
                    f"Cannot reach private IP {self.base_url} (timed out). "
                    "Cloud-hosted websites cannot directly access private home Wi-Fi IPs across local router firewalls. "
                    "For cloud usage: 1) Run a free public tunnel (e.g. 'ngrok http 192.168.1.50:80') and paste the public URL here, or "
                    "2) Run the dashboard locally on your laptop via 'run_local_network.bat' so both are on the same Wi-Fi."
                )
            return False, f"Cannot reach ESP32-CAM at {self.base_url}: {e.reason}. Check IP address and Wi-Fi connection."
        except Exception as e:
            return False, f"Camera test error: {str(e)}"

    def probe_endpoints(self) -> List[dict]:
        """
        Scans common ESP32-CAM endpoints to find active JPEG snapshot routes.
        """
        results = []
        for ep in KNOWN_CAMERA_ENDPOINTS:
            target_url = f"{self.base_url}{ep}"
            try:
                req = urllib.request.Request(target_url, headers={"User-Agent": "CropCare-AI-Client"})
                with urllib.request.urlopen(req, timeout=2) as res:
                    ct = res.headers.get_content_type()
                    data = res.read(1024)
                    has_jpeg = data.find(b"\xff\xd8") != -1 or "image" in ct
                    results.append({
                        "endpoint": ep,
                        "status": res.status,
                        "content_type": ct,
                        "has_jpeg": has_jpeg
                    })
            except Exception as e:
                results.append({
                    "endpoint": ep,
                    "status": "Error",
                    "content_type": str(e),
                    "has_jpeg": False
                })
        return results

    def capture_snapshot(self, custom_path: Optional[str] = None) -> Tuple[Optional[Image.Image], Optional[str]]:
        """
        Captures a single high-resolution still frame from the ESP32-CAM.
        Decodes the JPEG stream into a clean RGB PIL Image.
        If the primary endpoint fails with HTML or non-image data, attempts automatic
        fallback to alternate known camera endpoints.

        Returns:
            Tuple of (pil_image, error_message)
        """
        active_path = custom_path or self.capture_path
        target_url = f"{self.base_url}{active_path}"

        try:
            req = urllib.request.Request(
                target_url,
                headers={"User-Agent": "CropCare-AI-Client"}
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                if response.status != 200:
                    return None, f"ESP32-CAM returned HTTP {response.status} at {active_path}."

                img_bytes = response.read()

            image, decode_err = decode_image_payload(img_bytes)
            if image is not None:
                return image, None

            # If primary endpoint returned non-image (e.g. HTML or text), try fallback endpoints
            for alt_ep in KNOWN_CAMERA_ENDPOINTS:
                if alt_ep == active_path:
                    continue
                try:
                    alt_url = f"{self.base_url}{alt_ep}"
                    alt_req = urllib.request.Request(alt_url, headers={"User-Agent": "CropCare-AI-Client"})
                    with urllib.request.urlopen(alt_req, timeout=3) as alt_res:
                        if alt_res.status == 200:
                            alt_bytes = alt_res.read()
                            alt_img, alt_err = decode_image_payload(alt_bytes)
                            if alt_img is not None:
                                self.capture_path = alt_ep
                                return alt_img, None
                except Exception:
                    continue

            # Finally attempt extracting a frame directly from the live MJPEG stream
            stream_img, stream_err = self.capture_stream_frame()
            if stream_img is not None:
                return stream_img, None

            return None, decode_err or "Failed to decode valid image from ESP32-CAM."

        except urllib.error.HTTPError as e:
            return None, f"HTTP Error {e.code} at {target_url}: {e.reason}."
        except urllib.error.URLError as e:
            return None, (
                f"Cannot reach ESP32-CAM at {target_url}. "
                f"Ensure the camera board is powered on and connected to the same WiFi network. ({e.reason})"
            )
        except Exception as e:
            return None, f"Failed to acquire image from ESP32-CAM: {str(e)}"


def get_hardware_specs() -> dict:
    """Returns technical specs of the recommended ESP32-CAM hardware setup."""
    return {
        "microcontroller": "ESP32-S / ESP32-CAM (Ai-Thinker module)",
        "processor": "Dual-core Xtensa 32-bit LX6, up to 240 MHz",
        "camera_sensor": "OmniVision OV2640 (2 Megapixel, UXGA 1600x1200)",
        "memory": "520 KB SRAM + 4 MB external PSRAM",
        "wireless": "802.11 b/g/n Wi-Fi + Bluetooth 4.2 BR/EDR & BLE",
        "power_input": "5V / 2A via FTDI USB-to-UART / External step-down regulator",
        "communication_protocol": "HTTP GET /capture endpoint over local WiFi subnet",
        "inference_topology": "Edge Acquisition → LAN Stream → Host PC AI Inference"
    }
