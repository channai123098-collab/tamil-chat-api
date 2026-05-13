import { GoogleGenAI, Modality } from "@google/genai";

// Supports:
// 1. Replit AI Integration: AI_INTEGRATIONS_GEMINI_API_KEY + AI_INTEGRATIONS_GEMINI_BASE_URL
// 2. Direct API: GEMINI_API_KEYS (comma-separated) or GEMINI_API_KEY
const apiKey =
  process.env.AI_INTEGRATIONS_GEMINI_API_KEY ||
  (process.env.GEMINI_API_KEYS?.split(",")[0]?.trim()) ||
  process.env.GEMINI_API_KEY;

if (!apiKey) {
  throw new Error(
    "Gemini API key required. Set AI_INTEGRATIONS_GEMINI_API_KEY (Replit) or GEMINI_API_KEYS (direct).",
  );
}

const baseUrl = process.env.AI_INTEGRATIONS_GEMINI_BASE_URL;

export const ai = new GoogleGenAI({
  apiKey,
  ...(baseUrl ? { httpOptions: { apiVersion: "", baseUrl } } : {}),
});

export async function generateImage(
  prompt: string
): Promise<{ b64_json: string; mimeType: string }> {
  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash-image",
    contents: [{ role: "user", parts: [{ text: prompt }] }],
    config: {
      responseModalities: [Modality.TEXT, Modality.IMAGE],
    },
  });

  const candidate = response.candidates?.[0];
  const imagePart = candidate?.content?.parts?.find(
    (part: { inlineData?: { data?: string; mimeType?: string } }) => part.inlineData
  );

  if (!imagePart?.inlineData?.data) {
    throw new Error("No image data in response");
  }

  return {
    b64_json: imagePart.inlineData.data,
    mimeType: imagePart.inlineData.mimeType || "image/png",
  };
}
