# API Specification — Arabic Handwritten Letter AI Service

The inference backend exposes a stateless REST API designed to integrate with the client's educational web application and mini-game.

---

## Base URL
```text
http://localhost:8000
```

---

## 1. Health Check

```http
GET /health
```

### Response `200 OK`
```json
{
  "status": "ok"
}
```

---

## 2. Model Information

```http
GET /api/v1/model-info
```

### Response `200 OK`
```json
{
  "model": "arabic-character-classifier",
  "version": "0.1.0",
  "classes": 28,
  "target_letters": [
    "ا", "ب", "ت", "ث", "ج", "ح", "خ", "د", "ذ", "ر", "ز", "س", "ش", "ص",
    "ض", "ط", "ظ", "ع", "غ", "ف", "ق", "ك", "ل", "م", "ن", "ه", "و", "ي"
  ],
  "confidence_threshold": 0.70
}
```

---

## 3. Recognize Letter

```http
POST /api/v1/recognize-letter
Content-Type: multipart/form-data
```

### Request Parameters
| Parameter | Type | Required | Description |
| :--- | :--- | :---: | :--- |
| `image` | Binary File | Yes | Captured image of the handwritten Arabic letter (JPEG/PNG/WebP) |

### Response `200 OK` (Accepted)
```json
{
  "success": true,
  "prediction": {
    "letter": "ب",
    "class_id": 1,
    "confidence": 0.947,
    "accepted": true,
    "name_en": "baa",
    "name_ar": "باء"
  },
  "processing_time_ms": 38.4,
  "message": "Letter recognized successfully.",
  "top_candidates": [
    {"letter": "ب", "class_id": 1, "confidence": 0.947, "accepted": true, "name_en": "baa", "name_ar": "باء"},
    {"letter": "ت", "class_id": 2, "confidence": 0.031, "accepted": false, "name_en": "taa", "name_ar": "تاء"},
    {"letter": "ث", "class_id": 3, "confidence": 0.012, "accepted": false, "name_en": "thaa", "name_ar": "ثاء"}
  ]
}
```

### Response `200 OK` (Low Confidence / Rejected)
```json
{
  "success": true,
  "prediction": {
    "letter": "ث",
    "class_id": 3,
    "confidence": 0.52,
    "accepted": false,
    "name_en": "thaa",
    "name_ar": "ثاء"
  },
  "processing_time_ms": 35.1,
  "message": "Confidence below acceptance threshold (0.70). Please capture the letter again with clearer lighting."
}
```

---

## 4. Client Integration Example (JavaScript)

```javascript
const formData = new FormData();
formData.append("image", imageBlob, "capture.jpg");

const response = await fetch("http://localhost:8000/api/v1/recognize-letter", {
  method: "POST",
  body: formData
});

const result = await response.json();

if (result.success && result.prediction.accepted) {
  // Pass recognized letter to client's existing game engine
  game.handleArabicLetter(result.prediction.letter);
} else {
  // Prompt child to retry
  game.showRetryPrompt(result.message);
}
```
