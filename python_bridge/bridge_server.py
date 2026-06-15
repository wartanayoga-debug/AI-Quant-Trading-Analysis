import logging
import os
import sys
import warnings
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")

if sys.platform.startswith("win") and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    # Force DefaultEventLoopPolicy to prevent Uvicorn from reverting it
    # Note: set_event_loop_policy above is sufficient for this process;
    # avoid monkey-patching asyncio.DefaultEventLoopPolicy to prevent
    # compatibility issues with other libraries.

class WinError64Filter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage() if hasattr(record, "getMessage") else str(record.msg)
        if "The specified network name is no longer available" in msg or "WinError 64" in msg:
            return False
        if record.exc_info:
            _, exc_value, _ = record.exc_info
            if isinstance(exc_value, OSError) and getattr(exc_value, "winerror", None) == 64:
                return False
        return True

logging.getLogger("asyncio").addFilter(WinError64Filter())

logging.basicConfig(level=logging.INFO, format="[Bridge %(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _load_local_env() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, value)
    except Exception as exc:
        log.warning("Failed to load .env for bridge: %s", exc)


_load_local_env()

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    log.error("FastAPI stack missing. Run: pip install -r python_bridge/requirements.txt")
    sys.exit(1)


# =============================================================================
# App setup
# =============================================================================

app = FastAPI(title="QuantPrime Python Bridge", version="2.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8765",
        "http://127.0.0.1:8765",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Bridge-Key"],
)


# ─── API Key Authentication ─────────────────────────────────────────────────
_BRIDGE_API_KEY = os.environ.get("BRIDGE_API_KEY", "").strip()


@app.middleware("http")
async def bridge_auth_middleware(request, call_next):
    """Validate X-Bridge-Key header when BRIDGE_API_KEY is configured."""
    # Skip auth if no key is configured (backward compatible for local dev)
    if not _BRIDGE_API_KEY:
        return await call_next(request)

    # Allow health endpoint without auth for monitoring
    if request.url.path == "/health":
        provided = request.headers.get("X-Bridge-Key", "")
        if not provided:
            return await call_next(request)

    provided_key = request.headers.get("X-Bridge-Key", "")
    if provided_key != _BRIDGE_API_KEY:
        from starlette.responses import JSONResponse
        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid or missing X-Bridge-Key header."},
        )
    return await call_next(request)

# =============================================================================
# Environment helpers
# =============================================================================

def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return default if raw is None or raw.strip() == "" else raw.strip()


# =============================================================================
# Chronos loader
# =============================================================================

CHRONOS_AVAILABLE = False
CHRONOS_BACKEND = "disabled"
CHRONOS_ACTIVE_MODEL_ID: Optional[str] = None
CHRONOS_REQUESTED_MODEL_ID = _env_str("CHRONOS_MODEL_ID", "amazon/chronos-2")
CHRONOS_FALLBACK_MODEL_ID = _env_str("CHRONOS_FALLBACK_MODEL_ID", "amazon/chronos-t5-small")
CHRONOS_FALLBACK_ACTIVE = False
CHRONOS_FALLBACK_REASON: Optional[str] = None
CHRONOS_LOAD_ERROR: Optional[str] = None
CHRONOS_DEVICE = _env_str("CHRONOS_DEVICE", "cpu")
chronos_pipeline: Any = None
chronos_lock = asyncio.Lock()
MODEL_LOAD_STARTED = False
MODEL_LOAD_COMPLETE = False
MODEL_LOAD_EVENT = None
MODEL_LOAD_LOCK = asyncio.Lock()


def _load_chronos() -> None:
    global CHRONOS_AVAILABLE
    global CHRONOS_BACKEND
    global CHRONOS_ACTIVE_MODEL_ID
    global CHRONOS_FALLBACK_ACTIVE
    global CHRONOS_FALLBACK_REASON
    global CHRONOS_LOAD_ERROR
    global chronos_pipeline

    enable_chronos = _env_bool("ENABLE_CHRONOS", True)
    requested_backend = _env_str("CHRONOS_BACKEND", "auto").lower()

    if not enable_chronos or requested_backend == "disabled":
        CHRONOS_AVAILABLE = False
        CHRONOS_BACKEND = "disabled"
        CHRONOS_LOAD_ERROR = "Chronos disabled by ENABLE_CHRONOS=false or CHRONOS_BACKEND=disabled."
        log.info("Chronos disabled.")
        return

    if requested_backend in {"auto", "chronos2", "chronos-2", "chronos_bolt", "chronos-bolt", "bolt"}:
        try:
            model_id = CHRONOS_REQUESTED_MODEL_ID
            is_bolt_model = "bolt" in model_id.lower() or requested_backend in {"chronos_bolt", "chronos-bolt", "bolt"}
            if is_bolt_model:
                from chronos import ChronosBoltPipeline  # type: ignore

                pipeline_cls = ChronosBoltPipeline
                backend_name = "chronos_bolt"
                log.info("Loading Chronos-Bolt model %s on %s...", model_id, CHRONOS_DEVICE)
            else:
                from chronos import Chronos2Pipeline  # type: ignore

                pipeline_cls = Chronos2Pipeline
                backend_name = "chronos2"
                log.info("Loading Chronos-2 model %s on %s...", model_id, CHRONOS_DEVICE)

            chronos_pipeline = pipeline_cls.from_pretrained(
                model_id,
                device_map=CHRONOS_DEVICE,
            )

            CHRONOS_AVAILABLE = True
            CHRONOS_BACKEND = backend_name
            CHRONOS_ACTIVE_MODEL_ID = model_id
            CHRONOS_FALLBACK_ACTIVE = False
            CHRONOS_FALLBACK_REASON = None
            CHRONOS_LOAD_ERROR = None
            log.info("%s loaded successfully: %s", "Chronos-Bolt" if is_bolt_model else "Chronos-2", model_id)
            return
        except Exception as exc:
            CHRONOS_LOAD_ERROR = f"Chronos-2/Bolt load failed: {exc}"
            log.warning("%s", CHRONOS_LOAD_ERROR)

            if requested_backend in {"chronos2", "chronos-2", "chronos_bolt", "chronos-bolt", "bolt"}:
                CHRONOS_AVAILABLE = False
                CHRONOS_BACKEND = requested_backend
                CHRONOS_ACTIVE_MODEL_ID = None
                CHRONOS_FALLBACK_ACTIVE = False
                CHRONOS_FALLBACK_REASON = f"Fallback disabled because CHRONOS_BACKEND={requested_backend}."
                return

    if requested_backend in {"auto", "chronos_t5", "chronos-t5", "t5"}:
        try:
            from chronos import ChronosPipeline  # type: ignore
            import torch

            fallback_id = CHRONOS_FALLBACK_MODEL_ID

            dtype = torch.float32 if CHRONOS_DEVICE == "cpu" else torch.bfloat16

            log.info(
                "Loading fallback Chronos-T5 model %s on %s with dtype=%s...",
                fallback_id,
                CHRONOS_DEVICE,
                dtype,
            )

            chronos_pipeline = ChronosPipeline.from_pretrained(
                fallback_id,
                device_map=CHRONOS_DEVICE,
                dtype=dtype,
            )

            CHRONOS_AVAILABLE = True
            CHRONOS_BACKEND = "chronos_t5"
            CHRONOS_ACTIVE_MODEL_ID = fallback_id
            CHRONOS_FALLBACK_ACTIVE = True
            CHRONOS_FALLBACK_REASON = CHRONOS_LOAD_ERROR or "Chronos-2 was not requested."
            log.info("Chronos-T5 fallback loaded successfully: %s", fallback_id)
            return
        except ImportError as exc:
            CHRONOS_LOAD_ERROR = (
                f"Chronos unavailable. Install optional packages: "
                f"pip install -U torch chronos-forecasting pandas pyarrow. Detail: {exc}"
            )
            log.warning("%s", CHRONOS_LOAD_ERROR)
        except Exception as exc:
            CHRONOS_LOAD_ERROR = f"Chronos-T5 fallback load failed: {exc}"
            log.warning("%s", CHRONOS_LOAD_ERROR)

    CHRONOS_AVAILABLE = False
    CHRONOS_BACKEND = requested_backend
    CHRONOS_ACTIVE_MODEL_ID = None
    if CHRONOS_FALLBACK_REASON is None:
        CHRONOS_FALLBACK_REASON = CHRONOS_LOAD_ERROR


# =============================================================================
# Kronos loader
# =============================================================================

KRONOS_AVAILABLE = False
KRONOS_MODEL_ID = _env_str("KRONOS_MODEL_ID", "NeoQuasar/Kronos-small")
KRONOS_TOKENIZER_ID = _env_str("KRONOS_TOKENIZER_ID", "NeoQuasar/Kronos-Tokenizer-base")
KRONOS_MAX_CONTEXT = int(_env_str("KRONOS_MAX_CONTEXT", "512"))
KRONOS_DEVICE = _env_str("KRONOS_DEVICE", "cpu")
kronos_predictor: Any = None


def _load_kronos() -> None:
    global KRONOS_AVAILABLE, kronos_predictor
    if not _env_bool("ENABLE_KRONOS", True):
        log.info("Kronos disabled by ENABLE_KRONOS=false.")
        return
    try:
        try:
            from .kronos_model import Kronos, KronosTokenizer, KronosPredictor
        except ImportError:
            from kronos_model import Kronos, KronosTokenizer, KronosPredictor
        tokenizer = KronosTokenizer.from_pretrained(KRONOS_TOKENIZER_ID)
        model = Kronos.from_pretrained(KRONOS_MODEL_ID)
        kronos_predictor = KronosPredictor(model, tokenizer, max_context=KRONOS_MAX_CONTEXT, device=KRONOS_DEVICE)
        KRONOS_AVAILABLE = True
        log.info("Kronos loaded successfully: %s", KRONOS_MODEL_ID)
    except ImportError:
        log.warning(
            "Kronos unavailable. Install: pip install torch>=2.0.0 einops huggingface_hub safetensors"
        )
    except Exception as exc:
        log.warning("Kronos load failed: %s", exc)


# =============================================================================
# FinBERT sentiment loader
# =============================================================================

FINBERT_AVAILABLE = False
FINBERT_MODEL_ID = _env_str("FINBERT_MODEL_ID", "ProsusAI/finbert")
FINBERT_DEVICE = _env_str("FINBERT_DEVICE", "cpu")
FINBERT_LOAD_ERROR: Optional[str] = None
finbert_pipeline: Any = None
finbert_lock = asyncio.Lock()


def _load_finbert() -> None:
    global FINBERT_AVAILABLE, FINBERT_LOAD_ERROR, finbert_pipeline
    if not _env_bool("ENABLE_FINBERT", True):
        FINBERT_AVAILABLE = False
        FINBERT_LOAD_ERROR = "FinBERT disabled by ENABLE_FINBERT=false."
        log.info("FinBERT disabled by ENABLE_FINBERT=false.")
        return
    try:
        from transformers import pipeline

        device = -1
        if FINBERT_DEVICE not in {"", "cpu", "-1"}:
            try:
                device = int(FINBERT_DEVICE)
            except Exception:
                device = 0

        log.info("Loading FinBERT sentiment model %s on %s...", FINBERT_MODEL_ID, FINBERT_DEVICE)
        finbert_pipeline = pipeline(
            "text-classification",
            model=FINBERT_MODEL_ID,
            tokenizer=FINBERT_MODEL_ID,
            device=device,
            truncation=True,
        )
        FINBERT_AVAILABLE = True
        FINBERT_LOAD_ERROR = None
        log.info("FinBERT loaded successfully: %s", FINBERT_MODEL_ID)
    except Exception as exc:
        FINBERT_AVAILABLE = False
        FINBERT_LOAD_ERROR = str(exc)
        log.warning("FinBERT load failed: %s", exc)


# =============================================================================
# Optional model dependencies
# =============================================================================

HMM_AVAILABLE = False
try:
    from hmmlearn.hmm import GaussianHMM

    HMM_AVAILABLE = True
except ImportError:
    log.warning("hmmlearn unavailable. HMM endpoint will return 503.")

try:
    from scipy import stats  # noqa: F401

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# =============================================================================
# Pydantic models
# =============================================================================

class ForecastRequest(BaseModel):
    prices: List[float]
    steps: int = 5
    timestamps: Optional[List[float]] = None


class ForecastResponse(BaseModel):
    prices: List[float]
    lower_band: List[float]
    upper_band: List[float]
    model: str
    confidence_interval: float


class KronosForecastRequest(BaseModel):
    candles: List[Dict[str, float]]
    pred_len: int = 20
    sample_count: int = 20


class FinBERTSentimentRequest(BaseModel):
    headlines: List[str]
    ticker: Optional[str] = None


class FinBERTSentimentItem(BaseModel):
    text: str
    label: str
    score: float
    confidence: float


class FinBERTSentimentResponse(BaseModel):
    label: str
    sentiment_score: float
    confidence: float
    model: str
    items: List[FinBERTSentimentItem]


class RegimeRequest(BaseModel):
    prices: List[float]
    volumes: Optional[List[float]] = None
    n_components: int = 5


class RegimeResponse(BaseModel):
    current_state: int
    state_label: str
    transition_probs: List[List[float]]
    state_probabilities: List[float]
    all_states: List[int]
    regime_confidence: float


class VARRequest(BaseModel):
    returns_matrix: List[List[float]]
    weights: List[float]
    confidence: float = 0.95


class VARResponse(BaseModel):
    portfolio_var: float
    conditional_var: float
    portfolio_volatility: float
    beta_exposure: float
    diversification_ratio: float


# =============================================================================
# Forecast helpers
# =============================================================================

def _clean_prices(prices: List[float]) -> List[float]:
    arr = []
    for p in prices:
        try:
            v = float(p)
            if np.isfinite(v) and v > 0:
                arr.append(v)
        except Exception:
            continue
    return arr


def holt_winters_fallback(prices: List[float], steps: int) -> Dict[str, List[float]]:
    prices = _clean_prices(prices)
    steps = max(1, min(int(steps), 120))

    if len(prices) < 5:
        last = float(prices[-1] if prices else 0.0)
        return {"prices": [last] * steps, "lower": [last * 0.98] * steps, "upper": [last * 1.02] * steps}

    arr = np.asarray(prices, dtype=np.float64)
    alpha, beta, phi = 0.2, 0.1, 0.92
    best_rmse = np.inf
    split = max(3, int(len(arr) * 0.8))

    for a in [0.1, 0.2, 0.3]:
        for b in [0.05, 0.1, 0.2]:
            for p in [0.85, 0.92, 0.98]:
                train = arr[:split]
                level, trend = train[0], 0.0
                fitted = []
                for y in train[1:]:
                    prev_level, prev_trend = level, trend
                    level = a * y + (1 - a) * (prev_level + p * prev_trend)
                    trend = b * (level - prev_level) + (1 - b) * p * prev_trend
                    fitted.append(prev_level + p * prev_trend)
                if fitted:
                    rmse = float(np.sqrt(np.mean((train[1:] - np.asarray(fitted)) ** 2)))
                    if rmse < best_rmse:
                        best_rmse, alpha, beta, phi = rmse, a, b, p

    level, trend = arr[0], 0.0
    fitted = [arr[0]]
    for y in arr[1:]:
        prev_level, prev_trend = level, trend
        level = alpha * y + (1 - alpha) * (prev_level + phi * prev_trend)
        trend = beta * (level - prev_level) + (1 - beta) * phi * prev_trend
        fitted.append(prev_level + phi * prev_trend)

    residuals = arr - np.asarray(fitted)
    std = float(np.std(residuals)) if len(residuals) > 1 else 0.0
    forecasts, lowers, uppers = [], [], []

    denom = max(1e-9, 1 - phi)
    for h in range(1, steps + 1):
        phi_sum = phi * (1 - phi**h) / denom
        forecast = level + phi_sum * trend
        margin = std * np.sqrt(1 + h * 0.15)
        forecasts.append(float(round(forecast, 6)))
        lowers.append(float(round(forecast - margin, 6)))
        uppers.append(float(round(forecast + margin, 6)))

    return {"prices": forecasts, "lower": lowers, "upper": uppers}


def _finbert_label_to_score(label: str, confidence: float) -> float:
    value = str(label or "").strip().lower()
    if value == "positive":
        return confidence
    if value == "negative":
        return -confidence
    return 0.0


def _extract_prediction_columns(pred_df: Any, steps: int) -> Tuple[List[float], List[float], List[float]]:
    cols = {str(c): c for c in getattr(pred_df, "columns", [])}

    median_col = None
    for name in ["predictions", "prediction", "mean", "0.5", "0.50", "median"]:
        if name in cols:
            median_col = cols[name]
            break

    lower_col = None
    for name in ["0.1", "0.10", "p10", "q10", "lower"]:
        if name in cols:
            lower_col = cols[name]
            break

    upper_col = None
    for name in ["0.9", "0.90", "p90", "q90", "upper"]:
        if name in cols:
            upper_col = cols[name]
            break

    if median_col is None:
        numeric_cols = []
        for c in getattr(pred_df, "columns", []):
            try:
                if np.issubdtype(pred_df[c].dtype, np.number):
                    numeric_cols.append(c)
            except Exception:
                continue
        if not numeric_cols:
            raise ValueError(f"Cannot find numeric prediction columns in Chronos-2 output: {list(pred_df.columns)}")
        median_col = numeric_cols[0]

    median = [float(x) for x in pred_df[median_col].tail(steps).to_numpy()]

    if lower_col is not None:
        lower = [float(x) for x in pred_df[lower_col].tail(steps).to_numpy()]
    else:
        lower = median

    if upper_col is not None:
        upper = [float(x) for x in pred_df[upper_col].tail(steps).to_numpy()]
    else:
        upper = median

    return median, lower, upper


def _timeframe_delta_from_ms(times_ms: List[float]) -> pd.Timedelta:
    if len(times_ms) < 2:
        return pd.Timedelta(minutes=1)
    deltas = []
    for i in range(1, len(times_ms)):
        delta = float(times_ms[i]) - float(times_ms[i - 1])
        if np.isfinite(delta) and delta > 0:
            deltas.append(delta)
    if not deltas:
        return pd.Timedelta(minutes=1)
    median_ms = float(np.median(np.asarray(deltas, dtype=np.float64)))
    return pd.Timedelta(milliseconds=max(60_000.0, median_ms))


def _build_kronos_timestamps(candles: List[Dict[str, float]], pred_len: int) -> Tuple[pd.Series, pd.Series]:
    times_ms: List[float] = []
    for candle in candles:
        raw = candle.get("time")
        try:
            value = float(raw) if raw is not None else np.nan
        except Exception:
            value = np.nan
        if np.isfinite(value) and value > 0:
            times_ms.append(value)

    if len(times_ms) == len(candles):
        x_index = pd.to_datetime(times_ms, unit="ms", utc=True).tz_convert(None)
        step = _timeframe_delta_from_ms(times_ms)
        y_index = pd.date_range(
            start=x_index[-1] + step,
            periods=pred_len,
            freq=step,
        )
    else:
        x_index = pd.date_range(
            start="2000-01-01",
            periods=len(candles),
            freq="min",
        )
        y_index = pd.date_range(
            start=x_index[-1] + pd.Timedelta(minutes=1),
            periods=pred_len,
            freq="min",
        )

    return pd.Series(x_index), pd.Series(y_index)


def _build_chronos2_context_timestamps(price_count: int, timestamps: Optional[List[float]]) -> pd.DatetimeIndex:
    times_ms: List[float] = []
    for raw in timestamps or []:
        try:
            value = float(raw)
        except Exception:
            value = np.nan
        if not np.isfinite(value) or value <= 0:
            continue
        # Support epoch seconds from older callers while keeping milliseconds as the standard.
        times_ms.append(value * 1000 if value < 10_000_000_000 else value)

    if len(times_ms) == price_count and price_count >= 2:
        step = _timeframe_delta_from_ms(times_ms)
        end = pd.to_datetime(times_ms[-1], unit="ms", utc=True).tz_convert(None)
        return pd.date_range(end=end, periods=price_count, freq=step)

    return pd.date_range("2000-01-01", periods=price_count, freq="min")


async def _chronos2_forecast(prices: List[float], steps: int, timestamps: Optional[List[float]] = None) -> ForecastResponse:
    import pandas as pd

    context_df = pd.DataFrame(
        {
            "id": ["series_0"] * len(prices),
            "timestamp": _build_chronos2_context_timestamps(len(prices), timestamps),
            "target": prices,
        }
    )

    async with chronos_lock:
        pred_df = chronos_pipeline.predict_df(
            context_df,
            prediction_length=steps,
            quantile_levels=[0.1, 0.5, 0.9],
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )

    median, low, high = _extract_prediction_columns(pred_df, steps)
    return ForecastResponse(
        prices=[float(round(v, 6)) for v in median],
        lower_band=[float(round(v, 6)) for v in low],
        upper_band=[float(round(v, 6)) for v in high],
        model=f"{CHRONOS_BACKEND}:{CHRONOS_ACTIVE_MODEL_ID}",
        confidence_interval=0.8,
    )


async def _chronos_t5_forecast(prices: List[float], steps: int) -> ForecastResponse:
    import torch

    dtype = torch.float32 if CHRONOS_DEVICE == "cpu" else torch.bfloat16

    async with chronos_lock:
        context = torch.tensor(prices, dtype=dtype).unsqueeze(0)
        forecast = chronos_pipeline.predict(
            inputs=context,
            prediction_length=steps,
            num_samples=20,
        )

    samples = forecast[0].float().numpy()
    median = np.median(samples, axis=0)
    low = np.percentile(samples, 10, axis=0)
    high = np.percentile(samples, 90, axis=0)
    return ForecastResponse(
        prices=[float(round(v, 6)) for v in median],
        lower_band=[float(round(v, 6)) for v in low],
        upper_band=[float(round(v, 6)) for v in high],
        model=f"{CHRONOS_BACKEND}:{CHRONOS_ACTIVE_MODEL_ID}",
        confidence_interval=0.8,
    )


@app.post("/forecast", response_model=ForecastResponse)
async def forecast_endpoint(req: ForecastRequest):
    await ensure_models_loaded()
    prices = _clean_prices(req.prices)
    steps = max(1, min(int(req.steps), 120))

    if len(prices) < 5:
        raise HTTPException(status_code=400, detail="Minimum 5 valid positive price points required.")

    if CHRONOS_AVAILABLE and chronos_pipeline is not None:
        try:
            if CHRONOS_BACKEND in {"chronos2", "chronos_bolt"}:
                return await _chronos2_forecast(prices, steps, req.timestamps)
            if CHRONOS_BACKEND == "chronos_t5":
                return await _chronos_t5_forecast(prices, steps)
        except Exception as exc:
            log.warning("Chronos inference failed, using fallback: %s", exc)

    fallback = holt_winters_fallback(prices, steps)
    return ForecastResponse(
        prices=fallback["prices"],
        lower_band=fallback["lower"],
        upper_band=fallback["upper"],
        model="holt-winters-fallback",
        confidence_interval=0.68,
    )


@app.post("/forecast/kronos", response_model=ForecastResponse)
async def forecast_kronos_endpoint(req: KronosForecastRequest):
    await ensure_models_loaded()
    if not KRONOS_AVAILABLE or kronos_predictor is None:
        raise HTTPException(status_code=503, detail="Kronos not available.")

    if len(req.candles) < 10:
        raise HTTPException(status_code=400, detail="Minimum 10 candles required.")

    df = pd.DataFrame(req.candles)
    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")

    if "volume" not in df.columns:
        df["volume"] = 0.0
    if "amount" not in df.columns:
        df["amount"] = df["volume"] * df[["open", "high", "low", "close"]].mean(axis=1)

    lookback = min(len(df), KRONOS_MAX_CONTEXT)
    pred_len = max(1, min(int(req.pred_len), 120))

    x_df = df.iloc[-lookback:][["open", "high", "low", "close", "volume", "amount"]]
    x_timestamp_all, y_timestamp = _build_kronos_timestamps(req.candles, pred_len)
    x_timestamp = x_timestamp_all.iloc[-lookback:].reset_index(drop=True)

    try:
        pred_df = kronos_predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=0.8,
            top_p=0.95,
            sample_count=req.sample_count,
            verbose=False,
        )
    except Exception as exc:
        log.warning("Kronos inference failed: %s", exc)
        prices = [float(v) for v in df["close"].tail(max(10, min(len(df), 120))).to_numpy()]
        fallback = holt_winters_fallback(prices, pred_len)
        return ForecastResponse(
            prices=fallback["prices"],
            lower_band=fallback["lower"],
            upper_band=fallback["upper"],
            model=f"kronos-fallback:holt-winters:{KRONOS_MODEL_ID}",
            confidence_interval=0.68,
        )

    close_col = "close" if "close" in pred_df.columns else pred_df.columns[3]
    prices = [float(round(v, 6)) for v in pred_df[close_col].tail(pred_len).to_numpy()]
    low_col = "low" if "low" in pred_df.columns else pred_df.columns[2]
    high_col = "high" if "high" in pred_df.columns else pred_df.columns[1]
    lower = [float(round(v, 6)) for v in pred_df[low_col].tail(pred_len).to_numpy()]
    upper = [float(round(v, 6)) for v in pred_df[high_col].tail(pred_len).to_numpy()]

    return ForecastResponse(
        prices=prices,
        lower_band=lower,
        upper_band=upper,
        model=f"kronos:{KRONOS_MODEL_ID}",
        confidence_interval=0.8,
    )


# =============================================================================
# FinBERT sentiment endpoint
# =============================================================================

@app.post("/sentiment/finbert", response_model=FinBERTSentimentResponse)
async def sentiment_finbert_endpoint(req: FinBERTSentimentRequest):
    await ensure_models_loaded()
    if not FINBERT_AVAILABLE or finbert_pipeline is None:
        raise HTTPException(status_code=503, detail=FINBERT_LOAD_ERROR or "FinBERT not available.")

    headlines = []
    for headline in req.headlines or []:
        text = str(headline or "").strip()
        if text:
            headlines.append(text[:512])

    if not headlines:
        return FinBERTSentimentResponse(
            label="neutral",
            sentiment_score=0.0,
            confidence=0.0,
            model=f"finbert:{FINBERT_MODEL_ID}",
            items=[],
        )

    async with finbert_lock:
        raw = await asyncio.to_thread(
            finbert_pipeline,
            headlines,
            truncation=True,
            max_length=192,
        )

    items: List[FinBERTSentimentItem] = []
    weighted_score = 0.0
    confidence_sum = 0.0
    for text, result in zip(headlines, raw):
        row = result[0] if isinstance(result, list) and result else result
        label = str(row.get("label", "neutral")).lower()
        confidence = float(row.get("score", 0.0))
        score = _finbert_label_to_score(label, confidence)
        items.append(
            FinBERTSentimentItem(
                text=text,
                label=label,
                score=round(score, 4),
                confidence=round(confidence, 4),
            )
        )
        weighted_score += score * max(0.05, confidence)
        confidence_sum += max(0.05, confidence)

    aggregate = weighted_score / max(1e-9, confidence_sum)
    mean_confidence = confidence_sum / max(1, len(items))
    label = "positive" if aggregate > 0.18 else "negative" if aggregate < -0.18 else "neutral"
    return FinBERTSentimentResponse(
        label=label,
        sentiment_score=round(float(np.clip(aggregate, -1, 1)), 4),
        confidence=round(float(np.clip(mean_confidence, 0, 1)), 4),
        model=f"finbert:{FINBERT_MODEL_ID}",
        items=items,
    )


# =============================================================================
# HMM regime endpoint
# =============================================================================

REGIME_LABELS = [
    "Panic / Crash",
    "Trending Bearish",
    "Sideways / Accumulation",
    "High Volatility",
    "Trending Bullish",
]


@app.post("/regime", response_model=RegimeResponse)
async def regime_endpoint(req: RegimeRequest):
    if not HMM_AVAILABLE:
        raise HTTPException(status_code=503, detail="hmmlearn not available.")

    prices = np.asarray(_clean_prices(req.prices), dtype=np.float64)
    if len(prices) < 30:
        raise HTTPException(status_code=400, detail="Minimum 30 valid price points required.")

    log_returns = np.diff(np.log(np.maximum(prices, 1e-9)))
    rolling_vol = np.asarray([np.std(log_returns[max(0, i - 4): i + 1]) for i in range(len(log_returns))])
    features = np.column_stack([log_returns, rolling_vol])

    n_components = max(2, min(req.n_components, max(2, len(features) // 5)))
    model = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=100, random_state=42)
    model.fit(features)
    states = model.predict(features)
    posteriors = model.predict_proba(features)
    current_state = int(states[-1])
    means = model.means_[:, 0]
    rank = int(np.where(np.argsort(means) == current_state)[0][0])
    label = REGIME_LABELS[min(len(REGIME_LABELS) - 1, int(rank * len(REGIME_LABELS) / n_components))]

    return RegimeResponse(
        current_state=current_state,
        state_label=label,
        transition_probs=model.transmat_.tolist(),
        state_probabilities=posteriors[-1].tolist(),
        all_states=states.tolist(),
        regime_confidence=round(float(np.max(posteriors[-1])), 4),
    )


# =============================================================================
# Portfolio VaR
# =============================================================================

@app.post("/portfolio/var", response_model=VARResponse)
async def portfolio_var_endpoint(req: VARRequest):
    matrix = np.asarray(req.returns_matrix, dtype=np.float64)
    weights = np.asarray(req.weights, dtype=np.float64)

    if matrix.ndim != 2:
        raise HTTPException(status_code=400, detail="returns_matrix must be 2D.")
    if len(weights) != matrix.shape[0]:
        raise HTTPException(status_code=400, detail="weights length must match asset count.")
    if abs(float(weights.sum())) < 1e-9:
        raise HTTPException(status_code=400, detail="weights sum to zero.")

    weights = weights / weights.sum()
    portfolio_returns = weights @ matrix
    alpha = 1 - req.confidence
    var = float(np.percentile(portfolio_returns, alpha * 100))
    tail = portfolio_returns[portfolio_returns <= var]
    cvar = float(tail.mean()) if len(tail) else var

    port_vol = float(np.std(portfolio_returns) * np.sqrt(252))

    if matrix.shape[0] > 1:
        cov_pb = float(np.cov(portfolio_returns, matrix[0])[0, 1])
        var_b = float(np.var(matrix[0]))
        beta = cov_pb / var_b if var_b > 1e-12 else 1.0
    else:
        beta = 1.0

    individual_vols = np.std(matrix, axis=1) * np.sqrt(252)
    weighted_avg_vol = float(weights @ individual_vols)
    div_ratio = weighted_avg_vol / port_vol if port_vol > 1e-12 else 1.0

    return VARResponse(
        portfolio_var=round(var, 6),
        conditional_var=round(cvar, 6),
        portfolio_volatility=round(port_vol, 6),
        beta_exposure=round(float(beta), 4),
        diversification_ratio=round(float(div_ratio), 4),
    )


# =============================================================================
# Health
# =============================================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_load_started": MODEL_LOAD_STARTED,
        "model_load_complete": MODEL_LOAD_COMPLETE,
        "chronos_available": CHRONOS_AVAILABLE,
        "chronos_2_0": CHRONOS_AVAILABLE and CHRONOS_BACKEND in {"chronos2", "chronos_bolt"},
        "chronos_backend": CHRONOS_BACKEND,
        "chronos_model_id": CHRONOS_ACTIVE_MODEL_ID,
        "chronos_requested_model_id": CHRONOS_REQUESTED_MODEL_ID,
        "chronos_fallback_model_id": CHRONOS_FALLBACK_MODEL_ID,
        "fallback_active": CHRONOS_FALLBACK_ACTIVE,
        "fallback_reason": CHRONOS_FALLBACK_REASON,
        "load_error": CHRONOS_LOAD_ERROR,
        "device": CHRONOS_DEVICE,
        "hmm": HMM_AVAILABLE,
        "kronos_available": KRONOS_AVAILABLE,
        "kronos_model_id": KRONOS_MODEL_ID if KRONOS_AVAILABLE else None,
        "finbert_available": FINBERT_AVAILABLE,
        "finbert_model_id": FINBERT_MODEL_ID if FINBERT_AVAILABLE else None,
        "finbert_load_error": FINBERT_LOAD_ERROR,
        "scipy": SCIPY_AVAILABLE,
        "platform": sys.platform,
        "python": sys.version.split()[0],
    }


@app.on_event("startup")
async def startup_event():
    log.info(
        "Bridge HTTP ready. Models will lazy-load on first request. ChronosAvailable=%s Kronos=%s FinBERT=%s HMM=%s",
        CHRONOS_AVAILABLE,
        KRONOS_AVAILABLE,
        FINBERT_AVAILABLE,
        HMM_AVAILABLE,
    )

async def ensure_models_loaded():
    global MODEL_LOAD_STARTED, MODEL_LOAD_EVENT, MODEL_LOAD_COMPLETE
    
    async with MODEL_LOAD_LOCK:
        if MODEL_LOAD_EVENT is None:
            MODEL_LOAD_EVENT = asyncio.Event()

        if MODEL_LOAD_COMPLETE:
            return

        if not MODEL_LOAD_STARTED:
            MODEL_LOAD_STARTED = True
            should_load = True
        else:
            should_load = False
            
    if should_load:
        log.info("Lazy-loading models into VRAM...")
        await load_models_background()
    else:
        await MODEL_LOAD_EVENT.wait()


async def load_models_background():
    global MODEL_LOAD_COMPLETE
    try:
        # Chronos, Kronos, and FinBERT all touch shared torch/transformers/HF
        # loader state. Load them sequentially to avoid Windows lazy-import and
        # meta-tensor races during first initialization.
        await asyncio.to_thread(_load_chronos)
        await asyncio.to_thread(_load_kronos)
        await asyncio.to_thread(_load_finbert)
    finally:
        MODEL_LOAD_COMPLETE = True
        if MODEL_LOAD_EVENT is not None:
            MODEL_LOAD_EVENT.set()
        log.info(
            "Model loading complete. ChronosAvailable=%s Kronos=%s FinBERT=%s",
            CHRONOS_AVAILABLE,
            KRONOS_AVAILABLE,
            FINBERT_AVAILABLE,
        )


if __name__ == "__main__":
    port = int(os.environ.get("BRIDGE_PORT", 8765))
    uvicorn.run(app, host="127.0.0.1", port=port, reload=False, log_level="info")
