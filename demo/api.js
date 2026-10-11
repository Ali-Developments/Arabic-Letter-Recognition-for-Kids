/**
 * Arabic Handwriting AI - API Client Library
 * Implements the integration contract specified in ARABIC_HANDWRITING_AI_PROJECT_BRIEF.md.
 */

class ArabicRecognitionAPI {
  constructor(baseUrl = "") {
    // Priority order:
    // 1. Explicit constructor argument
    // 2. URL search parameter (?apiUrl=...)
    // 3. Global window.API_BASE_URL
    // 4. Current page origin (works automatically on both localhost and phone LAN/tunnel URLs)
    const urlParams = typeof window !== "undefined" && window.location ? new URLSearchParams(window.location.search) : null;
    const paramUrl = urlParams ? urlParams.get("apiUrl") : null;
    const globalUrl = typeof window !== "undefined" ? window.API_BASE_URL : null;
    const originUrl = typeof window !== "undefined" && window.location && window.location.origin && window.location.origin.includes("http") ? window.location.origin : "http://localhost:8000";

    this.baseUrl = baseUrl || paramUrl || globalUrl || originUrl;
  }

  /**
   * Health check
   */
  async checkHealth() {
    try {
      const res = await fetch(`${this.baseUrl}/health`);
      return await res.json();
    } catch (err) {
      console.error("Health check failed:", err);
      return { status: "offline", error: err.message };
    }
  }

  /**
   * Fetch model metadata
   */
  async getModelInfo() {
    try {
      const res = await fetch(`${this.baseUrl}/api/v1/model-info`);
      return await res.json();
    } catch (err) {
      console.error("Get model info failed:", err);
      return null;
    }
  }

  /**
   * Recognize an isolated handwritten Arabic letter from a Blob / File
   * @param {Blob|File} imageBlob - Binary image
   * @param {string} filename - Optional multipart filename
   * @returns {Promise<Object>} RecognitionResponse
   */
  async recognizeLetter(imageBlob, filename = "letter_capture.jpg") {
    if (!imageBlob || !(imageBlob instanceof Blob)) {
      throw new Error("Invalid image: expected a non-empty Blob or File instance.");
    }

    const formData = new FormData();
    formData.append("image", imageBlob, filename);

    const endpointUrl = `${this.baseUrl}/api/v1/recognize-letter`;
    console.log(`[API REQUEST] Sending ${imageBlob.size} bytes (${imageBlob.type}) to: ${endpointUrl}`);

    const response = await fetch(endpointUrl, {
      method: "POST",
      body: formData,
    });

    console.log(`[API RESPONSE] HTTP ${response.status} ${response.statusText}`);

    const rawText = await response.text();
    console.log("[API RAW RESPONSE]", rawText);

    let jsonData;
    try {
      jsonData = JSON.parse(rawText);
    } catch (parseErr) {
      console.error("Failed to parse JSON response:", parseErr, rawText);
      throw new Error(`Invalid JSON from server (HTTP ${response.status}): ${rawText.slice(0, 150)}`);
    }

    if (!response.ok) {
      throw new Error(jsonData.detail || `Server responded with status ${response.status}`);
    }

    return jsonData;
  }
}

// Global export for vanilla browser usage
window.ArabicRecognitionAPI = ArabicRecognitionAPI;
