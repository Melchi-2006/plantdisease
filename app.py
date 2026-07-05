import io
import os
from pathlib import Path

# Minimal fallback ASGI app so the module can be imported during deployment
# detection on platforms that don't have third-party packages installed yet.
async def _fallback_asgi_app(scope, receive, send):
    if scope.get("type") == "http":
        body = b'{"error":"dependencies not installed"}'
        headers = [(b"content-type", b"application/json")]
        await send({"type": "http.response.start", "status": 503, "headers": headers})
        await send({"type": "http.response.body", "body": body})
    else:
        await send({"type": "http.disconnect"})

app = _fallback_asgi_app

# Lazy-import optional dependencies (Pillow, Starlette) to avoid import-time
# failures during platform detection. If available, we'll overwrite `app`
# below with a proper Starlette instance.
try:
    from PIL import Image, ImageOps
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import HTMLResponse, JSONResponse
    from starlette.routing import Route
    STARLETTE_AVAILABLE = True
except Exception:
    Image = None
    ImageOps = None
    Starlette = None
    Request = None
    HTMLResponse = None
    JSONResponse = None
    Route = None
    STARLETTE_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent
CLASSIFIER_MODEL = BASE_DIR / "tomato_disease_model.pth"
CLASSES_PATH = BASE_DIR / "classes.pth"

MODEL = None
CLASSES = None
TRANSFORM = None
DEVICE = None

HTML_PAGE = """
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>Tomato Disease Detector</title>
  <style>
    body {font-family: Arial, sans-serif; margin: 0; background: #f8fff8; color: #123;}
    .wrap {max-width: 800px; margin: 40px auto; padding: 24px; background: white; border-radius: 16px; box-shadow: 0 8px 24px rgba(0,0,0,0.08);}
    h1 {color: #2a7f2a;}
    input[type=file] {margin: 12px 0;}
    button {background: #2a7f2a; color: white; border: none; padding: 10px 16px; border-radius: 8px; cursor: pointer;}
    .result {margin-top: 20px; padding: 12px; background: #f3fff3; border-radius: 8px;}
    img {max-width: 100%; margin-top: 12px; border-radius: 8px;}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <h1>🍅 Tomato Disease Detector</h1>
    <p>Upload a tomato leaf image to get a disease prediction.</p>
    <form id=\"predict-form\" enctype=\"multipart/form-data\">
      <input type=\"file\" name=\"image\" accept=\"image/*\" required>
      <br>
      <button type=\"submit\">Predict</button>
    </form>
    <div id=\"result\" class=\"result\">Waiting for an image...</div>
  </div>
  <script>
    const form = document.getElementById('predict-form');
    const result = document.getElementById('result');
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const formData = new FormData(form);
      result.innerHTML = 'Processing...';
      const response = await fetch('/predict', {method: 'POST', body: formData});
      const data = await response.json();
      if (data.error) {
        result.innerHTML = '<strong>Error:</strong> ' + data.error;
      } else {
        result.innerHTML = '<strong>Disease:</strong> ' + data.disease + '<br>' + '<strong>Confidence:</strong> ' + data.confidence + '%';
      }
    });
  </script>
</body>
</html>
"""


def load_models():
    global MODEL, CLASSES, TRANSFORM, DEVICE
    if MODEL is not None and CLASSES is not None and TRANSFORM is not None:
        return

    # Lazy-import heavy ML libraries so the module can be imported
    # on deployment platforms that don't have these packages available
    import torch
    import torch.nn as nn
    from torchvision import models, transforms

    if not CLASSIFIER_MODEL.exists() or not CLASSES_PATH.exists():
        raise FileNotFoundError("Model files not found in the project folder")

    CLASSES = torch.load(CLASSES_PATH, map_location="cpu")

    MODEL = models.efficientnet_b0(weights=None)
    MODEL.classifier[1] = nn.Linear(MODEL.classifier[1].in_features, len(CLASSES))
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL.load_state_dict(torch.load(CLASSIFIER_MODEL, map_location=DEVICE))
    MODEL.to(DEVICE)
    MODEL.eval()

    TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


async def home(request: Request) -> HTMLResponse:
    return HTMLResponse(HTML_PAGE)


async def predict(request: Request) -> JSONResponse:
    try:
        form = await request.form()
        image_file = form.get("image")
        if image_file is None:
            return JSONResponse({"error": "No image uploaded"}, status_code=400)

        load_models()
        # Ensure torch is available in this scope (lazy import)
        import torch

        image_bytes = await image_file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = ImageOps.exif_transpose(image)

        tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = MODEL(tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

        disease = CLASSES[pred.item()]
        confidence = round(conf.item() * 100, 2)

        return JSONResponse({
            "disease": disease,
            "confidence": confidence,
        })
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


if STARLETTE_AVAILABLE:
    app = Starlette(routes=[
    Route("/", home),
    Route("/predict", predict, methods=["POST"]),
])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)