import OpenAI from "openai";

// Optional — only required if OpenAI features are used
const apiKey = process.env.AI_INTEGRATIONS_OPENAI_API_KEY || process.env.OPENAI_API_KEY;
const baseURL = process.env.AI_INTEGRATIONS_OPENAI_BASE_URL;

export const openai = new OpenAI({
  apiKey: apiKey || "not-configured",
  ...(baseURL ? { baseURL } : {}),
});
