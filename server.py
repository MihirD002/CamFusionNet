#!/usr/bin/env python
"""
FAPNet Local Inference Server
Provides live prediction endpoint for the React UI.
Endpoints:
  GET  /api/health   -> Status check
  POST /api/predict  -> Runs FAPNet camouflage segmentation on uploaded image

Environment variables:
  MODEL_PATH    checkpoint to load (default: Snapshot/FAPNet/checkpoint_best.pth)
  MFAM_VARIANT  MFAM variant matching MODEL_PATH: original (default), depthwise,
                dilated, factorized, combined
"""

import os
import io
import json
import base64
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from lib.model import FAPNet, MFAM_VARIANTS
from utils.checkpoint import load_model_weights

# Device setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Snapshot/FAPNet/checkpoint_best.pth or Snapshot/FAPNet/FAPNet_best.pth matches the current ConvNeXt architecture
MODEL_PATH = os.environ.get('MODEL_PATH', os.path.join('Snapshot', 'FAPNet', 'checkpoint_best.pth'))
MFAM_VARIANT = os.environ.get('MFAM_VARIANT', 'original').strip().lower()
TEST_SIZE = 352

print(f"[Server] Initializing FAPNet ({MFAM_VARIANT} MFAM) on {DEVICE}...")
print(f"[Server] Loading checkpoint: {os.path.abspath(MODEL_PATH)}")
model = None
try:
    if MFAM_VARIANT not in MFAM_VARIANTS:
        raise ValueError(f"Unknown MFAM_VARIANT '{MFAM_VARIANT}'. Valid: {', '.join(MFAM_VARIANTS)}")
    if os.path.exists(MODEL_PATH):
        model = FAPNet(channel=64, mfam_variant=MFAM_VARIANT).to(DEVICE)

        # Handles full training checkpoints and raw state dicts, and raises a clear
        # error when the checkpoint was trained with a different MFAM variant.
        meta = load_model_weights(model, MODEL_PATH, MFAM_VARIANT, device=DEVICE)
        extra_info = []
        if 'epoch' in meta:
            extra_info.append(f"epoch: {meta['epoch']}")
        if 'best_mae' in meta:
            extra_info.append(f"best MAE: {meta['best_mae']:.4f}")
        info_str = f" ({', '.join(extra_info)})" if extra_info else ""

        model.eval()
        print(f"[Server] ✓ Model loaded successfully from {MODEL_PATH}{info_str}")
        print(f"[Server] ✓ Device: {DEVICE} | MFAM: {MFAM_VARIANT} | Parameters: {sum(p.numel() for p in model.parameters()):,}")
    else:
        print(f"[Server] ✗ Checkpoint not found at: {os.path.abspath(MODEL_PATH)}")
        print(f"[Server]   Running in canvas-fallback mode (no neural inference).")
except Exception as e:
    model = None
    print(f"[Server] ✗ Error loading model: {e}")
    print(f"[Server]   Note: set MFAM_VARIANT to the variant the checkpoint was trained with; "
          f"older checkpoints (e.g. checkpoints/FAPNet.pth) must also match the backbone architecture.")


# Preprocessing transform
img_transform = transforms.Compose([
    transforms.Resize((TEST_SIZE, TEST_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class InferenceHandler(BaseHTTPRequestHandler):
    def _send_cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        if self.path == '/api/health':
            self.send_response(200)
            self._send_cors_headers()
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response = {
                'status': 'online',
                'model_loaded': model is not None,
                'mfam_variant': MFAM_VARIANT,
                'device': str(DEVICE),
                'cuda_available': torch.cuda.is_available()
            }
            self.wfile.write(json.dumps(response).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/api/predict':
            start_time = time.time()
            content_length = int(self.headers.get('Content-Length', 0))
            raw_body = self.rfile.read(content_length)

            try:
                raw_image_bytes = None

                # Check if JSON
                try:
                    body_json = json.loads(raw_body.decode('utf-8'))
                    if 'image_base64' in body_json:
                        b64_str = body_json['image_base64']
                        if ',' in b64_str:
                            b64_str = b64_str.split(',', 1)[1]
                        raw_image_bytes = base64.b64decode(b64_str)
                except Exception:
                    # If not JSON, assume raw image bytes directly
                    raw_image_bytes = raw_body

                if not raw_image_bytes:
                    self.send_response(400)
                    self._send_cors_headers()
                    self.end_headers()
                    self.wfile.write(b'{"error": "No image data received"}')
                    return

                # Open and process image
                pil_img = Image.open(io.BytesIO(raw_image_bytes)).convert('RGB')
                orig_w, orig_h = pil_img.size

                if model is not None:
                    # Run PyTorch model
                    tensor_img = img_transform(pil_img).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        _, _, _, cam, _ = model(tensor_img)
                        res = F.interpolate(cam, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
                        res = res.sigmoid().data.cpu().numpy().squeeze()
                        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                        mask_np = (res * 255).astype(np.uint8)
                else:
                    np_img = np.array(pil_img)
                    gray = np_img.mean(axis=2).astype(np.uint8)
                    mask_np = 255 - gray

                # Encode output mask to PNG in base64
                mask_pil = Image.fromarray(mask_np)
                buffer = io.BytesIO()
                mask_pil.save(buffer, format="PNG")
                mask_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

                elapsed_ms = round((time.time() - start_time) * 1000, 1)

                self.send_response(200)
                self._send_cors_headers()
                self.send_header('Content-Type', 'application/json')
                self.end_headers()

                res_obj = {
                    'status': 'success',
                    'inference_time_ms': elapsed_ms,
                    'device': str(DEVICE),
                    'mfam_variant': MFAM_VARIANT,
                    'width': orig_w,
                    'height': orig_h,
                    'mask_base64': mask_b64
                }
                self.wfile.write(json.dumps(res_obj).encode('utf-8'))

            except Exception as e:
                print(f"[Server] Inference error: {e}")
                self.send_response(500)
                self._send_cors_headers()
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()


def run(port=8000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, InferenceHandler)
    print(f"[Server] FAPNet Inference Server active at http://127.0.0.1:{port}")
    httpd.serve_forever()


if __name__ == '__main__':
    run()
