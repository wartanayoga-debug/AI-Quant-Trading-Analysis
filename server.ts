import dotenv from "dotenv";
import express, { NextFunction, Request, Response } from "express";
import helmet from "helmet";
import path from "path";
import { createServer as createViteServer } from "vite";
import { createServer } from "http";
import { Server } from "socket.io";
import apiRouter from "./src/server/routes/api";
import simulatorRouter from "./src/server/routes/simulator.routes";
import playbookRouter from "./src/server/routes/playbook.routes";
import chatRouter from "./src/server/routes/chat.routes";
import { checkBridgeHealth } from "./src/server/engines/bridge_client";
import { EventBus } from "./src/server/engines/event_bus";
import { LMStudioClient } from "./src/server/engines/lm_studio";
import { MarketDataEngine } from "./src/server/engines/data.engine";
import { runMigrations } from "./src/server/utils/migrations";
import { PlaybookEngine } from "./src/server/engines/playbook.engine";
import { BackgroundScanner } from "./src/server/engines/background_scanner";
import { TelegramBotEngine } from "./src/server/engines/telegram/telegram_bot";
import { VirtualLearningOrchestrator } from "./src/server/engines/learning/virtual_learning_orchestrator";

dotenv.config();

const PORT = Number(process.env.PORT || 3000);
const HOST = process.env.HOST || "127.0.0.1";
const ALLOWED_ORIGINS = [
  `http://localhost:${PORT}`,
  `http://127.0.0.1:${PORT}`,
  "http://localhost:3000",
  "http://127.0.0.1:3000",
];
const heavyEndpointHits = new Map<string, { count: number; resetAt: number }>();
// Periodically clean up expired rate limit entries to prevent memory leak
setInterval(() => {
  const now = Date.now();
  for (const [key, entry] of heavyEndpointHits) {
    if (entry.resetAt <= now) heavyEndpointHits.delete(key);
  }
}, 300_000).unref(); // every 5 minutes, doesn't keep process alive

function heavyEndpointRateLimit(req: Request, res: Response, next: NextFunction) {
  const heavy =
    req.path === "/scan" ||
    req.path === "/screener" ||
    req.path === "/signals/generate" ||
    req.path === "/signals/stability-test" ||
    req.path === "/risk-of-ruin" ||
    req.path === "/scanner/pairs" ||
    req.path === "/ml/pipeline" ||
    req.path === "/ml/train" ||
    req.path === "/ml/backtest" ||
    req.path === "/v6/ai-signals/generate" ||
    req.path === "/v6/analysis/signal-stability-test" ||
    req.path === "/v3/signal" ||
    req.path === "/v3/screener" ||
    req.path === "/v5/risk-of-ruin" ||
    req.path.startsWith("/autogluon") ||
    req.path.startsWith("/ai-signal-pipeline");
  if (!heavy) return next();

  const windowMs = Number(process.env.API_HEAVY_RATE_LIMIT_WINDOW_MS || 60_000);
  const maxRequests = Number(process.env.API_HEAVY_RATE_LIMIT_MAX || 60);
  const key = `${req.ip || "local"}:${req.path}`;
  const now = Date.now();
  const current = heavyEndpointHits.get(key);
  if (!current || current.resetAt <= now) {
    heavyEndpointHits.set(key, { count: 1, resetAt: now + windowMs });
    return next();
  }

  current.count += 1;
  if (current.count > maxRequests) {
    return res.status(429).json({
      success: false,
      ok: false,
      error_code: "RATE_LIMITED",
      message: "Too many heavy API requests. Please retry shortly.",
      data_status: "provider_error",
      retry_after_ms: Math.max(0, current.resetAt - now),
    });
  }

  return next();
}

async function runStartupChecks() {
  try {
    console.log("[Server] Running startup checks...");
    const lm = LMStudioClient.getInstance();
    const lmStatus = await lm.healthCheck();
    console.log(`  LM Studio: ${lmStatus.available ? "Online" : "Offline (fallback active)"}`);

    const bridge = await checkBridgeHealth().catch(() => null);
    console.log(`  Python Bridge: ${bridge ? "Online" : "Offline (fallback active)"}`);

    const dataEngine = MarketDataEngine.getInstance();
    const candles = await dataEngine.getHistory("BBRI.JK", "IDX", "1d", 5).catch(() => []);
    console.log(`  Yahoo Finance: ${candles.length > 0 ? "Online" : "Offline - check internet connection"}`);

    void dataEngine.warmCacheForTickers(
      ["BBRI.JK", "BBCA.JK", "BMRI.JK", "BBNI.JK", "TLKM.JK"],
      "IDX",
      "15m",
      120,
    );
    void dataEngine.warmCacheForTickers(
      ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"],
      "CRYPTO",
      "15m",
      120,
    );
  } catch (err: any) {
    console.warn(`[Server] Startup checks partially failed: ${err?.message || err}`);
  }
}

async function startServer() {
  runMigrations();
  PlaybookEngine.getInstance().ensureDefaults();
  const app = express();
  const httpServer = createServer(app);
  const io = new Server(httpServer, {
    cors: {
      origin: ALLOWED_ORIGINS,
      credentials: true,
    },
  });
  const bus = EventBus.getInstance();

  io.on("connection", (socket) => {
    socket.emit("system_status", {
      status: "WS_CONNECTED",
      message: "QuantPrime Stream Ready",
    });
  });

  bus.on("scan_requested", (evt) => {
    io.emit("live_event", { type: "SCAN_START", data: evt });
  });
  bus.on("scan_progress", (evt) => {
    io.emit("live_event", { type: "SCAN_PROGRESS", data: evt });
  });
  bus.on("scan_completed", (evt) => {
    io.emit("live_event", { type: "SCAN_COMPLETE", data: evt });
  });
  bus.on("alert_triggered", (evt) => {
    io.emit("price_alert", evt);
  });
  bus.on("ai_signal_created", (evt) => {
    io.emit("ai_signal", { type: "AI_SIGNAL_CREATED", data: evt });
  });
  bus.on("ai_signal_updated", (evt) => {
    io.emit("ai_signal", { type: "AI_SIGNAL_UPDATED", data: evt });
  });

  app.use(
    helmet({
      contentSecurityPolicy: false,
      crossOriginEmbedderPolicy: false,
    }),
  );
  app.use(express.json({ limit: "2mb" }));
  app.use(express.urlencoded({ extended: true }));

  app.use((req, res, next) => {
    const start = Date.now();
    res.on("finish", () => {
      const ms = Date.now() - start;
      if (ms > 2000) {
        console.log(`[API] ${req.method} ${req.path} - ${res.statusCode} (${ms}ms)`);
      }
    });
    next();
  });

  app.use("/api", heavyEndpointRateLimit, apiRouter);
  app.use("/api/simulator", simulatorRouter);
  app.use("/api/playbooks", playbookRouter);
  app.use("/api/chat", chatRouter);
  app.get(["/health", "/api/health"], (_req, res) => {
    res.json({
      status: "healthy",
      timestamp: new Date().toISOString(),
      platform: "IDX & Crypto AI Trading Analysis Platform",
    });
  });
  app.use("/api", (req, res) => {
    res.status(404).json({
      success: false,
      ok: false,
      error_code: "API_ROUTE_NOT_FOUND",
      message: `API route not found: ${req.method} ${req.originalUrl}`,
      data_status: "provider_error",
    });
  });

  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (_req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.use((err: Error & { status?: number; type?: string }, _req: Request, res: Response, _next: NextFunction) => {
    const status = err.status === 400 || err.type === "entity.parse.failed" ? 400 : 500;
    const errorCode = status === 400 ? "INVALID_JSON_BODY" : "INTERNAL_SERVER_ERROR";
    console.error("[Server] Unhandled error:", err.message);
    res.status(status).json({
      success: false,
      ok: false,
      error_code: errorCode,
      message: status === 400
        ? "Request body must be valid JSON."
        : process.env.NODE_ENV === "production" ? "Internal server error" : err.message,
      data_status: "provider_error",
    });
  });

  httpServer.listen(PORT, HOST, async () => {
    console.log(`[Server] Listening on http://${HOST}:${PORT}`);
    await runStartupChecks();
    TelegramBotEngine.getInstance();
    BackgroundScanner.getInstance().start();
    VirtualLearningOrchestrator.getInstance().start();
  });

  const shutdown = () => {
    console.log("[Server] Graceful shutdown initiated...");
    httpServer.close(() => {
      console.log("[Server] Closed.");
      process.exit(0);
    });
    setTimeout(() => process.exit(1), 6000);
  };
  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
}

startServer().catch((error) => {
  console.error("[Server] Fatal bootstrap exception:", error);
  process.exit(1);
});
