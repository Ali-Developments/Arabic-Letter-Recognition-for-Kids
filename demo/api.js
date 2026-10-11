/**
 * Arabic Handwriting AI - API Client Library
 * Implements the integration contract specified in ARABIC_HANDWRITING_AI_PROJECT_BRIEF.md.
 */

class ArabicRecognitionAPI {
  constructor(baseUrl = "") {
    // If baseUrl is empty, default to current origin or http://localhost:8000
    this.baseUrl = baseUrl || (window.location.origin.includes("http") ? window.location.origin : "http://localhost:8000");
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
   * @returns {Promise<Object>} RecognitionResponse
   */
  async recognizeLetter(imageBlob) {
    const formData = new FormData();
    formData.append("image", imageBlob, "letter_capture.jpg");

    const response = await fetch(`${this.baseUrl}/api/v1/recognize-letter`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Server responded with status ${response.status}`);
    }

    return await response.json();
  }
}

// Global export for vanilla browser usage
window.ArabicRecognitionAPI = ArabicRecognitionAPI;
