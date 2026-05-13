import { GoogleGenAI } from "@google/genai";

// ── Multi-key rotation support ───────────────────────────────────────────────
// Supports comma-separated keys: GEMINI_API_KEYS="key1,key2,key3"
// Also supports single key: GEMINI_API_KEY="key1"
// Replit integration: AI_INTEGRATIONS_GEMINI_API_KEY (single key from Replit)

function parseKeys(): string[] {
  const replitKey = process.env.AI_INTEGRATIONS_GEMINI_API_KEY;
  const multiKeys = process.env.GEMINI_API_KEYS;
  const singleKey = process.env.GEMINI_API_KEY;

  if (replitKey) return [replitKey];
  if (multiKeys) {
    const keys = multiKeys.split(",").map((k) => k.trim()).filter(Boolean);
    if (keys.length > 0) return keys;
  }
  if (singleKey) return [singleKey];
  return [];
}

const ALL_KEYS = parseKeys();

if (ALL_KEYS.length === 0) {
  throw new Error(
    "Gemini API key is required. Set GEMINI_API_KEYS (comma-separated for rotation), " +
    "GEMINI_API_KEY (single key), or AI_INTEGRATIONS_GEMINI_API_KEY (Replit integration).",
  );
}

let currentKeyIndex = 0;

function getNextKey(): string {
  const key = ALL_KEYS[currentKeyIndex % ALL_KEYS.length]!;
  currentKeyIndex = (currentKeyIndex + 1) % ALL_KEYS.length;
  return key;
}

function getCurrentKey(): string {
  return ALL_KEYS[currentKeyIndex % ALL_KEYS.length]!;
}

const baseUrl = process.env.AI_INTEGRATIONS_GEMINI_BASE_URL;

function makeClient(apiKey: string): GoogleGenAI {
  return new GoogleGenAI({
    apiKey,
    ...(baseUrl
      ? { httpOptions: { apiVersion: "", baseUrl } }
      : {}),
  });
}

// ── Proxy client that auto-rotates keys on 429 / 401 / 403 ──────────────────
class RotatingGeminiClient {
  private clients: GoogleGenAI[];
  private clientIndex = 0;

  constructor(keys: string[]) {
    this.clients = keys.map(makeClient);
  }

  get models() {
    const self = this;
    return {
      async generateContent(params: Parameters<GoogleGenAI["models"]["generateContent"]>[0]) {
        let lastError: Error = new Error("No keys available");
        for (let attempt = 0; attempt < self.clients.length; attempt++) {
          const client = self.clients[self.clientIndex % self.clients.length]!;
          try {
            return await client.models.generateContent(params);
          } catch (err) {
            lastError = err instanceof Error ? err : new Error(String(err));
            const msg = lastError.message.toLowerCase();
            const isKeyError =
              msg.includes("401") || msg.includes("403") ||
              msg.includes("invalid") || msg.includes("unauthorized") ||
              msg.includes("permission") || msg.includes("api_key") ||
              msg.includes("429") || msg.includes("rate") || msg.includes("quota") ||
              msg.includes("resource_exhausted");
            if (isKeyError && self.clients.length > 1) {
              self.clientIndex = (self.clientIndex + 1) % self.clients.length;
              continue;
            }
            throw err;
          }
        }
        throw lastError;
      },

      async *generateContentStream(params: Parameters<GoogleGenAI["models"]["generateContentStream"]>[0]) {
        let lastError: Error = new Error("No keys available");
        for (let attempt = 0; attempt < self.clients.length; attempt++) {
          const client = self.clients[self.clientIndex % self.clients.length]!;
          try {
            const stream = await client.models.generateContentStream(params);
            yield* stream;
            return;
          } catch (err) {
            lastError = err instanceof Error ? err : new Error(String(err));
            const msg = lastError.message.toLowerCase();
            const isKeyError =
              msg.includes("401") || msg.includes("403") ||
              msg.includes("invalid") || msg.includes("unauthorized") ||
              msg.includes("permission") || msg.includes("api_key") ||
              msg.includes("429") || msg.includes("rate") || msg.includes("quota") ||
              msg.includes("resource_exhausted");
            if (isKeyError && self.clients.length > 1) {
              self.clientIndex = (self.clientIndex + 1) % self.clients.length;
              continue;
            }
            throw err;
          }
        }
        throw lastError;
      },
    };
  }
}

export const ai = new RotatingGeminiClient(ALL_KEYS) as unknown as GoogleGenAI;

export { getCurrentKey, getNextKey };
