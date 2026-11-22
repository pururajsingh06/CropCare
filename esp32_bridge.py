"""
CropCare AI - ESP32-CAM Dual-Port Bridge
Merges ESP32-CAM Port 80 (Still Capture) and Port 81 (MJPEG Live Stream)
onto a single local port (8080), allowing BOTH services to run through
a SINGLE free ngrok tunnel!

Usage:
    python esp32_bridge.py [ESP32_IP_ADDRESS]

Example:
    python esp32_bridge.py 192.168.1.50

Then run ngrok in another terminal:
    ngrok http 8080
"""

import sys
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.request
import urllib.error
import socket

DEFAULT_ESP32_IP = "192.168.1.50"
HTTP_PORT = 80
STREAM_PORT = 81
BRIDGE_PORT = 8080

target_ip = DEFAULT_ESP32_IP
target_http_port = HTTP_PORT
target_stream_port = STREAM_PORT


class DualPortBridgeHandler(BaseHTTPRequestHandler):
    """
    Proxies HTTP requests based on path:
      - /stream* -> ESP32:81/stream (MJPEG live video)
      - everything else (/capture, /, etc.) -> ESP32:80 (HTTP still snapshot & controls)
    """

    def do_GET(self):
        is_stream = self.path.startswith("/stream") or ":81" in self.path
        target_port = target_stream_port if is_stream else target_http_port
        target_url = f"http://{target_ip}:{target_port}{self.path}"

        try:
            req = urllib.request.Request(
                target_url,
                headers={
                    "User-Agent": "CropCare-Bridge",
                    "Accept": "*/*"
                }
            )

            # Open upstream connection to ESP32-CAM
            with urllib.request.urlopen(req, timeout=12) as upstream_resp:
                self.send_response(upstream_resp.status)

                # Relay upstream headers
                for header_key, header_val in upstream_resp.headers.items():
                    # Filter hop-by-hop headers
                    if header_key.lower() not in ["transfer-encoding", "connection", "server"]:
                        self.send_header(header_key, header_val)

                # Add CORS headers so web browsers & Streamlit Cloud can stream without security blocks
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "*")
                self.end_headers()

                # Stream response bytes back to client
                while True:
                    chunk = upstream_resp.read(16384)
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        # Client disconnected (e.g. paused stream)
                        break

        except urllib.error.HTTPError as e:
            self.send_error(e.code, f"ESP32-CAM HTTP Error: {e.reason}")
        except urllib.error.URLError as e:
            self.send_error(502, f"Bridge cannot reach ESP32-CAM at {target_url} ({e.reason})")
        except Exception as e:
            self.send_error(500, f"Bridge internal error: {str(e)}")

    def do_OPTIONS(self):
        # Handle CORS preflight
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def log_message(self, format, *args):
        # Silence routine 200 stream logging to keep console clean
        if "200" not in args[1]:
            super().log_message(format, *args)


def main():
    global target_ip, target_http_port, target_stream_port

    parser = argparse.ArgumentParser(description="CropCare AI - ESP32-CAM Dual-Port Bridge")
    parser.add_argument("ip", nargs="?", default=DEFAULT_ESP32_IP, help="IP address of your ESP32-CAM (default: 192.168.1.50)")
    parser.add_argument("--port", type=int, default=BRIDGE_PORT, help=f"Local bridge listening port (default: {BRIDGE_PORT})")
    parser.add_argument("--http-port", type=int, default=HTTP_PORT, help=f"ESP32 HTTP port (default: {HTTP_PORT})")
    parser.add_argument("--stream-port", type=int, default=STREAM_PORT, help=f"ESP32 MJPEG stream port (default: {STREAM_PORT})")
    args = parser.parse_args()

    target_ip = args.ip.strip()
    target_http_port = args.http_port
    target_stream_port = args.stream_port
    listen_port = args.port

    server = HTTPServer(("0.0.0.0", listen_port), DualPortBridgeHandler)
    server.timeout = 5

    print("=" * 65)
    print("🌿 CropCare AI – ESP32-CAM Dual-Port Bridge")
    print("=" * 65)
    print(f"Target ESP32-CAM:   http://{target_ip}")
    print(f"  • Still Capture:  http://{target_ip}:{target_http_port}/capture")
    print(f"  • Live Stream:    http://{target_ip}:{target_stream_port}/stream")
    print(f"Bridge Active on:   http://localhost:{listen_port}")
    print("-" * 65)
    print("⚡ HOW TO EXPOSE OVER NGROK (Single Free Tunnel):")
    print(f"  In a new terminal window, run:")
    print(f"      ngrok http {listen_port}")
    print()
    print("  Then copy the resulting ngrok URL (e.g. https://xxx.ngrok-free.app)")
    print("  and paste it into the 'ESP32-CAM IP or Public Tunnel URL' box in")
    print("  your Streamlit dashboard!")
    print("=" * 65)
    print(f"Listening on port {listen_port}... Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nBridge stopped cleanly.")
        server.server_close()


if __name__ == "__main__":
    main()
