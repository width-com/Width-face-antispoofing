# FLIP Face Anti-Spoofing API Documentation

## Overview

Face liveness detection service based on FLIP (Cross-domain Face Anti-spoofing with Language Guidance). Uses 3 pre-trained models (CeFA, WMCA, SURF) with majority voting for robust spoof detection.

## Base URL

```
Production: https://<your-domain>.trycloudflare.com
Local:      http://localhost:5010
```

---

## POST /predict

Detect whether a face image is real (live) or spoofed (attack).

### Request

Support one of the following inputs:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| image | File | No | Image file (JPEG/PNG/BMP/WebP), any resolution |
| image_base64 | String | No | Base64 image bytes, supports plain base64 or `data:image/...;base64,...` |
| s3_uri | String | No | S3 URI such as `s3://bucket/path/to/image.jpg` |

Exactly one input source is required.

**Content-Type:** `multipart/form-data` or `application/json`

### Response

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "label": "real",
    "score": 0.876543,
    "cefa_score": 0.831200,
    "wmca_score": 0.912345,
    "surf_score": 0.886084
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| code | int | 200=success, 400=bad request, 500=server error |
| message | string | Status description |
| data.label | string | `"real"` or `"spoof"` (majority vote of 3 models) |
| data.score | float | Average real-probability across 3 models (0~1) |
| data.cefa_score | float | CeFA model real-probability (0~1) |
| data.wmca_score | float | WMCA model real-probability (0~1) |
| data.surf_score | float | SURF model real-probability (0~1) |

> **Interpretation:** `score` closer to 1 = more likely real, closer to 0 = more likely spoof. `label` is determined by majority vote (2 out of 3 models agree).

### Error Response

```json
{
  "code": 400,
  "message": "Invalid image: Failed to decode image",
  "data": null
}
```

---

## Code Examples

### cURL

```bash
curl -X POST https://<BASE_URL>/predict \
  -F "image=@/path/to/face.jpg"
```

### Python (requests)

```python
import requests

url = "https://<BASE_URL>/predict"

with open("face.jpg", "rb") as f:
    resp = requests.post(url, files={"image": f})

result = resp.json()
if result["code"] == 200:
    data = result["data"]
    print(f"Label: {data['label']}")
    print(f"Score: {data['score']:.4f}")
    print(f"CeFA: {data['cefa_score']:.4f}")
    print(f"WMCA: {data['wmca_score']:.4f}")
    print(f"SURF: {data['surf_score']:.4f}")
```

### Python (base64)

```python
import base64
import requests

url = "https://<BASE_URL>/predict"

with open("face.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

resp = requests.post(url, json={"image_base64": image_base64})
print(resp.json())
```

### Python (S3 URI)

```python
import requests

url = "https://<BASE_URL>/predict"
resp = requests.post(url, json={"s3_uri": "s3://workflow-kyc/kyc_ocr_front.jpg"})
print(resp.json())
```

### Python (aiohttp)

```python
import aiohttp
import asyncio

async def predict(image_path: str):
    url = "https://<BASE_URL>/predict"
    async with aiohttp.ClientSession() as session:
        with open(image_path, "rb") as f:
            data = aiohttp.FormData()
            data.add_field("image", f, filename="face.jpg")
            async with session.post(url, data=data) as resp:
                return await resp.json()

result = asyncio.run(predict("face.jpg"))
print(result)
```

### JavaScript (fetch)

```javascript
const formData = new FormData();
formData.append("image", fileInput.files[0]);

const resp = await fetch("https://<BASE_URL>/predict", {
  method: "POST",
  body: formData,
});
const result = await resp.json();

if (result.code === 200) {
  console.log("Label:", result.data.label);
  console.log("Score:", result.data.score);
}
```

### Java (OkHttp)

```java
OkHttpClient client = new OkHttpClient();

RequestBody body = new MultipartBody.Builder()
    .setType(MultipartBody.FORM)
    .addFormDataPart("image", "face.jpg",
        RequestBody.create(new File("face.jpg"), MediaType.parse("image/jpeg")))
    .build();

Request request = new Request.Builder()
    .url("https://<BASE_URL>/predict")
    .post(body)
    .build();

Response response = client.newCall(request).execute();
System.out.println(response.body().string());
```

### Go

```go
package main

import (
    "bytes"
    "fmt"
    "io"
    "mime/multipart"
    "net/http"
    "os"
)

func main() {
    file, _ := os.Open("face.jpg")
    defer file.Close()

    var buf bytes.Buffer
    writer := multipart.NewWriter(&buf)
    part, _ := writer.CreateFormFile("image", "face.jpg")
    io.Copy(part, file)
    writer.Close()

    resp, _ := http.Post(
        "https://<BASE_URL>/predict",
        writer.FormDataContentType(),
        &buf,
    )
    defer resp.Body.Close()
    body, _ := io.ReadAll(resp.Body)
    fmt.Println(string(body))
}
```

---

## Notes

- Input images are resized to 224x224 internally; any resolution is accepted.
- Typical latency: **~50ms** (local GPU), **~1.5s** (public via Cloudflare Tunnel).
- The 3 models share the same CLIP ViT-B/16 backbone but are trained on different datasets, providing cross-domain robustness.
- CORS is enabled; the API can be called directly from browsers.
