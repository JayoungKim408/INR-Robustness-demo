"""Live mode: serves web/ and runs the real attack on any digit you draw in the browser.

    python demo/serve.py                  # http://127.0.0.1:8000   (GPU if available; CUDA_VISIBLE_DEVICES= for CPU)
    python demo/serve.py --port 8123

The page detects this server (/api/health) and enables the "Draw your own" panel. Without it, the page still works
from the precomputed web/data/sweep.json.
"""
import argparse
import base64
import json
import os
import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import ASSETS, DEVICE, VICTIMS, Engine  # noqa: E402

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
LOCK = threading.Lock()   # one attack at a time
ENGINE = None


def b64(t01):
    return base64.b64encode((t01.clamp(0, 1).numpy() * 255).round().astype(np.uint8).tobytes()).decode()


def attack_request(pixels, eps):
    x = (torch.tensor(pixels, dtype=torch.float32).view(1, 1, 28, 28).clamp(0, 1) - 0.5) / 0.5
    x = x.to(DEVICE)
    with LOCK:
        clean_inr = ENGINE.fit(x)
        out = {"input": b64((x[0, 0].cpu() + 1) / 2), "victims": {}}
        for name, title in VICTIMS.items():
            with torch.no_grad():
                y = ENGINE.victims[name](clean_inr).argmax(1)   # no label for a drawn digit: attack the clean prediction
            r = ENGINE.evaluate(name, x, y, eps, clean_inr)
            out["victims"][name] = dict(
                title=title, probs_clean=r["probs_clean"][0].tolist(), probs_adv=r["probs_adv"][0].tolist(),
                adv=b64((r["x_adv"][0, 0] + 1) / 2), rec_clean=b64(r["rec_clean"][0]), rec_adv=b64(r["rec_adv"][0]))
    return out


class Handler(SimpleHTTPRequestHandler):
    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/health":
            return self._json(200, {"live": True, "device": DEVICE})
        super().do_GET()

    def do_POST(self):
        if self.path != "/api/attack":
            return self._json(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            if n > 100_000:
                return self._json(413, {"error": "payload too large"})
            req = json.loads(self.rfile.read(n))
            pixels = [float(p) for p in req["pixels"]]
            assert len(pixels) == 784
            eps = min(max(float(req.get("eps", 0.3)), 0.01), 0.5)
            self._json(200, attack_request(pixels, eps))
        except Exception as e:  # noqa: BLE001
            self._json(400, {"error": str(e)})

    def log_message(self, fmt, *args):
        sys.stderr.write("[serve] " + fmt % args + "\n")


def main():
    global ENGINE
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("DEMO_THREADS", 16)))
    ENGINE = Engine(torch.load(f"{ASSETS}/samples.pt", weights_only=False))
    print(f"device={DEVICE}; open http://{args.host}:{args.port}")
    ThreadingHTTPServer((args.host, args.port), partial(Handler, directory=WEB)).serve_forever()


if __name__ == "__main__":
    main()
