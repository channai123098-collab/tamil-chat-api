import { GoogleGenAI } from "@google/genai";

// Supports two modes:
// 1. Replit AI Integrations: AI_INTEGRATIONS_GEMINI_API_KEY + AI_INTEGRATIONS_GEMINI_BASE_URL
// 2. Direct Google AI Studio: GEMINI_API_KEY (for Render.com / self-hosted)
const apiKey =
  process.env.AI_INTEGRATIONS_GEMINI_API_KEY || process.env.GEMINI_API_KEY;

if (!apiKey) {
  throw new Error(
    "Gemini API key is required. Set AI_INTEGRATIONS_GEMINI_API_KEY (Replit) or GEMINI_API_KEY (Render / self-hosted).",
  );
}

const baseUrl = process.env.AI_INTEGRATIONS_GEMINI_BASE_URL;

export const ai = new GoogleGenAI({
  apiKey,
  ...(baseUrl
    ? { httpOptions: { apiVersion: "", baseUrl } }
    : {}),
});
