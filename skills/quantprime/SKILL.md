# QuantPrime V6 Agent Skill

QuantPrime V6 is a local manual-trading decision-support platform. It never executes real broker orders.

## Safety Rules

1. Never claim guaranteed profit.
2. Always preserve no-trade reasons.
3. Always show model status, data quality, expected value after cost, R:R, validation score, and signal freshness.
4. Always state: manual execution required.
5. Never convert an AI Signal into real broker execution.
6. For IDX bearish signals, use avoid-long / exit-reduce guidance, not default short-selling.

## Primary V6 Endpoints

### AI Signals
- `POST /api/v6/ai-signals/generate`
- `GET /api/v6/ai-signals`
- `GET /api/v6/ai-signals/:id`
- `POST /api/v6/ai-signals/:id/refresh`
- `POST /api/v6/ai-signals/:id/send-to-paper`
- `POST /api/v6/ai-signals/:id/send-to-simulator`
- `POST /api/v6/ai-signals/:id/mark-executed`
- `POST /api/v6/ai-signals/:id/reject`
- `POST /api/v6/ai-signals/resolve-outcomes`
- `GET /api/v6/ai-signals/performance`
- `POST /api/v6/ai-signals/degrade-check`

### Validation and Health
- `POST /api/v6/analysis/signal-stability-test`
- `GET /api/v6/provider-health`
- `GET /api/v6/calibration/status`
- `GET /api/v6/agent/model-health`
- `GET /api/v6/agent/provider-status`
- `GET /api/v6/agent/capabilities`

### Local AI Report (LM Studio)
- `GET /api/v6/lmstudio/health`
- `POST /api/v6/lmstudio/explain-signal`
- `POST /api/v6/lmstudio/generate-report`

### Governance
- `GET /api/v6/risk/signal-limits`
- `PATCH /api/v6/risk/signal-limits`
- `GET /api/v6/portfolio/conflicts`
- `GET /api/v6/portfolio/risk`

## Required Interpretation

If `status` is not `ACTIVE`, do not call it a trade. Treat it as watch-only.
If `noTradeReasons` contains blockers, the correct action is no-trade.
If `modelStatus` is not `VALIDATED`, treat ML probability as secondary confirmation only.
V5 routes are legacy compatibility only.
LM Studio reports must be grounded on structured backend metrics; no fabricated price/news narratives.
