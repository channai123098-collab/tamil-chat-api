import { Router, type IRouter, type Request, type Response } from "express";
import { MsEdgeTTS, OUTPUT_FORMAT } from "msedge-tts";
import { logger } from "../lib/logger";

const router: IRouter = Router();

interface TtsRequestBody {
  text?: string;
  voice?: string;
}

router.post("/tts", async (req: Request, res: Response) => {
  const body = req.body as TtsRequestBody;
  const text = typeof body.text === "string" ? body.text.trim() : "";
  const voice = typeof body.voice === "string" ? body.voice : "ta-IN-PallaviNeural";

  if (!text) {
    res.status(400).json({ error: "text is required" });
    return;
  }

  if (text.length > 5000) {
    res.status(400).json({ error: "text too long (max 5000 chars)" });
    return;
  }

  try {
    const tts = new MsEdgeTTS();
    await tts.setMetadata(voice, OUTPUT_FORMAT.AUDIO_24KHZ_48KBITRATE_MONO_MP3);

    const readable = tts.toStream(text);
    const chunks: Buffer[] = [];

    await new Promise<void>((resolve, reject) => {
      readable.on("data", (chunk: Buffer) => {
        chunks.push(chunk);
      });
      readable.on("end", () => resolve());
      readable.on("error", (err: Error) => reject(err));
    });

    const audioBuffer = Buffer.concat(chunks);
    const base64Audio = audioBuffer.toString("base64");

    res.json({
      audio: base64Audio,
      mimeType: "audio/mpeg",
    });
  } catch (err) {
    logger.error({ err }, "TTS error");
    const message = err instanceof Error ? err.message : "TTS failed";
    res.status(500).json({ error: message });
  }
});

export default router;
