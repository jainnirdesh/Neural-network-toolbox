import { GoogleGenAI } from "@google/genai";

function getAiClient() {
  const apiKey = import.meta.env.VITE_GEMINI_API_KEY;
  if (!apiKey) {
    return null;
  }
  return new GoogleGenAI({ apiKey });
}

function localSentimentAnalysis(text: string) {
  const positiveWords = ["outstanding", "excellent", "great", "good", "amazing", "love", "happy", "efficient", "wonderful", "fantastic"];
  const negativeWords = ["bad", "poor", "terrible", "awful", "hate", "slow", "worst", "disappointing", "broken", "ugly"];
  const normalizedText = text.toLowerCase();

  let positiveScore = 0;
  let negativeScore = 0;

  for (const word of positiveWords) {
    if (normalizedText.includes(word)) positiveScore += 1;
  }

  for (const word of negativeWords) {
    if (normalizedText.includes(word)) negativeScore += 1;
  }

  if (!positiveScore && !negativeScore) {
    return {
      sentiment: "neutral",
      score: 0.5,
      reason: "No strong sentiment keywords were found, so the statement reads as neutral.",
    };
  }

  const totalScore = positiveScore + negativeScore;

  if (positiveScore > negativeScore) {
    return {
      sentiment: "positive",
      score: Math.max(0.55, positiveScore / totalScore),
      reason: "The statement contains positive language that outweighs any negative cues.",
    };
  }

  if (negativeScore > positiveScore) {
    return {
      sentiment: "negative",
      score: Math.max(0.55, negativeScore / totalScore),
      reason: "The statement contains negative language that outweighs any positive cues.",
    };
  }

  return {
    sentiment: "neutral",
    score: 0.5,
    reason: "The statement has mixed cues, so the overall tone is balanced.",
  };
}

export async function classifySentiment(text: string) {
  const fallback = localSentimentAnalysis(text);
  const ai = getAiClient();

  if (!ai) {
    return fallback;
  }

  try {
    const response = await ai.models.generateContent({
      model: "gemini-3-flash-preview",
      contents: `Analyze the sentiment of this text: "${text}". 
      Reply with ONLY one of these JSON fields: { "sentiment": "positive" | "negative" | "neutral", "score": 0-1, "reason": "brief reason" }`,
      config: {
        responseMimeType: "application/json",
      }
    });

    const parsed = JSON.parse(response.text || "{}");
    if (!parsed.sentiment || typeof parsed.score !== "number" || !parsed.reason) {
      return fallback;
    }

    return parsed;
  } catch (error) {
    console.warn("Gemini sentiment analysis failed, using local fallback.", error);
    return fallback;
  }
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
