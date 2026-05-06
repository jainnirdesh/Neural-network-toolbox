import { GoogleGenAI } from "@google/genai";

function getAiClient() {
  const apiKey = import.meta.env.VITE_GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error("Missing Gemini API key. Set VITE_GEMINI_API_KEY in your .env file.");
  }
  return new GoogleGenAI({ apiKey });
}

export async function classifySentiment(text: string) {
  const ai = getAiClient();
  const response = await ai.models.generateContent({
    model: "gemini-3-flash-preview",
    contents: `Analyze the sentiment of this text: "${text}". 
    Reply with ONLY one of these JSON fields: { "sentiment": "positive" | "negative" | "neutral", "score": 0-1, "reason": "brief reason" }`,
    config: {
      responseMimeType: "application/json",
    }
  });
  
  return JSON.parse(response.text || "{}");
}

export async function classifyImage(base64Image: string) {
  const ai = getAiClient();
  const response = await ai.models.generateContent({
    model: "gemini-3-flash-preview",
    contents: [
      { text: "Is this image a dog or a cat? Reply in JSON: { \"prediction\": \"cat\" | \"dog\", \"confidence\": 0-1 }" },
      { inlineData: { mimeType: "image/jpeg", data: base64Image.split(',')[1] } }
    ],
    config: {
      responseMimeType: "application/json",
    }
  });
  
  return JSON.parse(response.text || "{}");
}
