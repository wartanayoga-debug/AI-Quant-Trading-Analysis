import fs from "node:fs";
import path from "node:path";

import { MarketDataEngine } from "../src/server/engines/data.engine";
import { onlyClosedCandles } from "../src/server/utils/candle";
import { COMPREHENSIVE_ASSETS } from "../src/server/utils/assets";
import type { AssetClass, Candle } from "../src/types";

type RegistryAsset = (typeof COMPREHENSIVE_ASSETS)[number];

interface Options {
  out: string;
  markets: AssetClass[];
  symbols: string[];
  timeframes: string[];
  limit: number;
  maxSymbols: number;
  concurrency: number;
  delayMs: number;
}

const DEFAULT_CRYPTO_SYMBOLS = [
  "BTCUSDT",
  "ETHUSDT",
  "BNBUSDT",
  "SOLUSDT",
  "XRPUSDT",
  "ADAUSDT",
  "DOGEUSDT",
  "AVAXUSDT",
  "DOTUSDT",
  "LINKUSDT",
  "NEARUSDT",
  "UNIUSDT",
  "LTCUSDT",
  "ATOMUSDT",
  "APTUSDT",
  "SUIUSDT",
  "OPUSDT",
  "ARBUSDT",
  "INJUSDT",
  "AAVEUSDT",
];

const SUPPORTED_TIMEFRAMES = new Set(["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1D", "4d", "1W"]);

function parseList(raw: string | undefined): string[] {
  if (!raw) return [];
  return raw
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function parseArgs(argv: string[]): Options {
  const raw: Record<string, string> = {};
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (!token.startsWith("--")) continue;
    const key = token.slice(2);
    const next = argv[i + 1];
    if (!next || next.startsWith("--")) {
      raw[key] = "true";
    } else {
      raw[key] = next;
      i += 1;
    }
  }

  if (raw.help === "true" || raw.h === "true") {
    console.log(`Usage:
  npm run export:autogluon-data -- --out data/training/ohlcv.csv
  npm run export:autogluon-data -- --symbols "BTCUSDT,ETHUSDT,BBCA.JK" --timeframes "1h,4h,1d" --limit 800

Options:
  --out <path>              Output CSV path. Default: data/training/ohlcv.csv
  --markets <list>          CRYPTO, IDX, or CRYPTO,IDX. Default: CRYPTO
  --symbols <list>          Comma-separated tickers. Overrides default universe.
  --timeframes <list>       Comma-separated timeframes. Default: 1h,4h,1d
  --limit <n>               Candles per symbol/timeframe. Default: 800
  --max-symbols <n>         Safety cap when symbols are not provided. Default: 20
  --concurrency <n>         Parallel fetches. Default: 3
  --delay-ms <n>            Delay before each task starts. Default: 250`);
    process.exit(0);
  }

  const markets = parseList(raw.markets || raw.market)
    .map((item) => item.toUpperCase())
    .filter((item): item is AssetClass => item === "CRYPTO" || item === "IDX");
  const timeframes = parseList(raw.timeframes || raw.tf).length
    ? parseList(raw.timeframes || raw.tf)
    : ["1h", "4h", "1d"];
  const invalidTimeframes = timeframes.filter((timeframe) => !SUPPORTED_TIMEFRAMES.has(timeframe));
  if (invalidTimeframes.length) {
    throw new Error(
      `Unsupported timeframe(s): ${invalidTimeframes.join(",")}. ` +
      `Supported: ${[...SUPPORTED_TIMEFRAMES].join(",")}. ` +
      `In PowerShell, quote comma-separated values, for example --timeframes "1h,4h,1d".`,
    );
  }

  return {
    out: raw.out || "data/training/ohlcv.csv",
    markets: markets.length ? markets : ["CRYPTO"],
    symbols: parseList(raw.symbols || raw.tickers).map((item) => item.toUpperCase()),
    timeframes,
    limit: Math.max(30, Number(raw.limit || 800)),
    maxSymbols: Math.max(1, Number(raw["max-symbols"] || raw.maxSymbols || 20)),
    concurrency: Math.max(1, Math.min(8, Number(raw.concurrency || 3))),
    delayMs: Math.max(0, Number(raw["delay-ms"] || raw.delayMs || 250)),
  };
}

function registryByTicker(): Map<string, RegistryAsset> {
  const map = new Map<string, RegistryAsset>();
  for (const asset of COMPREHENSIVE_ASSETS) {
    const key = asset.ticker.toUpperCase();
    if (!map.has(key)) map.set(key, asset);
  }
  return map;
}

function inferAsset(ticker: string, registry: Map<string, RegistryAsset>): RegistryAsset {
  const normalized = ticker.toUpperCase();
  const found = registry.get(normalized);
  if (found) return found;
  return {
    ticker: normalized,
    name: normalized,
    assetClass: normalized.endsWith(".JK") ? "IDX" : "CRYPTO",
    sector: "Custom",
  };
}

function resolveAssets(options: Options): RegistryAsset[] {
  const registry = registryByTicker();
  if (options.symbols.length) {
    const seen = new Set<string>();
    return options.symbols
      .map((ticker) => inferAsset(ticker, registry))
      .filter((asset) => {
        const key = asset.ticker.toUpperCase();
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      });
  }

  const defaultSymbols = [
    ...(options.markets.includes("CRYPTO") ? DEFAULT_CRYPTO_SYMBOLS : []),
    ...(options.markets.includes("IDX")
      ? COMPREHENSIVE_ASSETS
        .filter((asset) => asset.assetClass === "IDX")
        .map((asset) => asset.ticker)
      : []),
  ];
  return defaultSymbols
    .map((ticker) => inferAsset(ticker, registry))
    .filter((asset) => options.markets.includes(asset.assetClass))
    .slice(0, options.maxSymbols);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function validCandle(candle: Candle): boolean {
  return (
    Number.isFinite(candle.time) &&
    Number.isFinite(candle.open) &&
    Number.isFinite(candle.high) &&
    Number.isFinite(candle.low) &&
    Number.isFinite(candle.close) &&
    Number.isFinite(candle.volume) &&
    candle.high >= candle.low &&
    candle.close > 0
  );
}

function csvEscape(value: string | number): string {
  const text = String(value);
  return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function toRows(asset: RegistryAsset, timeframe: string, candles: Candle[]): Array<string[]> {
  return candles.filter(validCandle).map((candle) => [
    new Date(candle.time).toISOString(),
    asset.ticker.toUpperCase(),
    asset.assetClass,
    timeframe,
    String(candle.open),
    String(candle.high),
    String(candle.low),
    String(candle.close),
    String(Math.max(0, candle.volume)),
  ]);
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const assets = resolveAssets(options);
  const dataEngine = MarketDataEngine.getInstance();
  const tasks = assets.flatMap((asset) => options.timeframes.map((timeframe) => ({ asset, timeframe })));
  const allRows: Array<string[]> = [];
  const failures: string[] = [];
  let cursor = 0;

  console.log(`[Export] assets=${assets.length} timeframes=${options.timeframes.join(",")} limit=${options.limit} out=${options.out}`);

  const worker = async (workerId: number) => {
    while (true) {
      const index = cursor;
      cursor += 1;
      if (index >= tasks.length) return;

      const { asset, timeframe } = tasks[index];
      const effectiveLimit = asset.assetClass === "CRYPTO" ? Math.min(options.limit, 1000) : options.limit;
      await sleep(options.delayMs * workerId);

      try {
        const candles = onlyClosedCandles(
          await dataEngine.getHistory(asset.ticker, asset.assetClass, timeframe, effectiveLimit),
        );
        const rows = toRows(asset, timeframe, candles);
        if (rows.length === 0) {
          failures.push(`${asset.ticker} ${timeframe}: no closed candles returned`);
        }
        allRows.push(...rows);
        console.log(`[Export] ${asset.ticker} ${timeframe}: ${rows.length} candles`);
      } catch (error: any) {
        const message = `${asset.ticker} ${timeframe}: ${error?.message || String(error)}`;
        failures.push(message);
        console.warn(`[Export] skipped ${message}`);
      }
    }
  };

  await Promise.all(
    Array.from({ length: Math.min(options.concurrency, tasks.length) }, (_, index) => worker(index)),
  );

  allRows.sort((a, b) =>
    a[1].localeCompare(b[1]) ||
    a[3].localeCompare(b[3]) ||
    a[0].localeCompare(b[0]),
  );

  if (!allRows.length) {
    throw new Error("No OHLCV rows exported. Check provider connectivity, symbols, markets, and timeframe choices.");
  }

  const header = ["timestamp", "symbol", "market", "timeframe", "open", "high", "low", "close", "volume"];
  const csv = [header, ...allRows]
    .map((row) => row.map(csvEscape).join(","))
    .join("\n") + "\n";

  fs.mkdirSync(path.dirname(options.out), { recursive: true });
  fs.writeFileSync(options.out, csv, "utf8");

  console.log(`[Export] wrote ${allRows.length} rows to ${options.out}`);
  if (failures.length) {
    console.warn(`[Export] skipped ${failures.length} failed fetches. First failures:`);
    for (const failure of failures.slice(0, 8)) console.warn(`  - ${failure}`);
  }
}

main().catch((error) => {
  console.error(`[Export] failed: ${error?.message || String(error)}`);
  process.exitCode = 1;
});
