import { Router, type IRouter } from "express";
import healthRouter from "./health";
import chatRouter from "./chat";
import imageRouter from "./image";
import ttsRouter from "./tts";

const router: IRouter = Router();

router.use(healthRouter);
router.use(chatRouter);
router.use(imageRouter);
router.use(ttsRouter);

export default router;
