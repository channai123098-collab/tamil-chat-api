import { Router, type IRouter, type Request, type Response } from "express";
import PERSONA_REGISTRY from "../data/persona-registry.js";

const router: IRouter = Router();

interface PersonaEntry {
  id: string;
  name: string;
  custom?: boolean;
}

const BUILTIN_PERSONAS: PersonaEntry[] = PERSONA_REGISTRY;

const customPersonas = new Map<string, PersonaEntry>();

router.get("/personas", (_req: Request, res: Response) => {
  const custom = [...customPersonas.values()];
  res.json({ personas: [...BUILTIN_PERSONAS, ...custom] });
});

router.post("/personas", (req: Request, res: Response) => {
  const { id, name } = req.body as { id?: string; name?: string };
  if (!id || !name || typeof id !== "string" || typeof name !== "string") {
    res.status(400).json({ error: "id and name required" }); return;
  }
  if (BUILTIN_PERSONAS.some(p => p.id === id)) {
    res.json({ ok: true, skipped: "builtin" }); return;
  }
  customPersonas.set(id, { id, name, custom: true });
  res.json({ ok: true });
});

router.post("/personas/bulk", (req: Request, res: Response) => {
  const list = req.body as { id: string; name: string }[];
  if (!Array.isArray(list)) { res.status(400).json({ error: "array required" }); return; }
  for (const { id, name } of list) {
    if (!id || !name) continue;
    if (BUILTIN_PERSONAS.some(p => p.id === id)) continue;
    customPersonas.set(id, { id, name, custom: true });
  }
  res.json({ ok: true, count: list.length });
});

router.delete("/personas/:id", (req: Request, res: Response) => {
  const id = req.params["id"];
  if (!id) { res.status(400).json({ error: "id required" }); return; }
  customPersonas.delete(id);
  res.json({ ok: true });
});

export default router;
