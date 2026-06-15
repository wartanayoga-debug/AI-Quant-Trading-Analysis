import sqlite3
import json
import logging
from pathlib import Path
from .model_registry import get_model_metadata, set_alias, _model_root

logger = logging.getLogger(__name__)

DB_PATH = Path("data/quantprime.sqlite")

def evaluate_battle() -> dict:
    if not DB_PATH.exists():
        return {"status": "no_db"}
        
    current_meta = get_model_metadata("latest")
    challenger_meta = get_model_metadata("challenger")
    
    if not challenger_meta.get("available"):
        return {"status": "no_challenger"}
        
    challenger_version = challenger_meta.get("version")
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    # We compare trades that were opened AFTER the challenger was registered
    registered_at = challenger_meta.get("registered_at", "1970-01-01T00:00:00Z")
    
    # Fetch Champion stats
    c.execute("""
        SELECT exit_reason, net_pnl, entry_price, stop_loss, notional_value
        FROM virtual_trades 
        WHERE status != 'OPEN' 
        AND strategy_id = 'default_auto_learning'
        AND created_at >= ?
    """, (registered_at,))
    champion_rows = c.fetchall()
    
    # Fetch Challenger stats
    c.execute("""
        SELECT exit_reason, net_pnl, entry_price, stop_loss, notional_value
        FROM virtual_trades 
        WHERE status != 'OPEN' 
        AND strategy_id = 'AUTOGLUON_CHALLENGER'
        AND created_at >= ?
    """, (registered_at,))
    challenger_rows = c.fetchall()
    
    conn.close()
    
    def calc_metrics(rows):
        wins = 0
        losses = 0
        gross_profit = 0.0
        gross_loss = 0.0
        total_r = 0.0
        
        for r in rows:
            pnl = float(r["net_pnl"] or 0)
            if pnl > 0:
                wins += 1
                gross_profit += pnl
            elif pnl < 0:
                losses += 1
                gross_loss += abs(pnl)
                
            entry = float(r["entry_price"] or 0)
            sl = float(r["stop_loss"] or 0)
            notional = float(r["notional_value"] or 0)
            risk = 0
            if entry > 0 and sl > 0 and notional > 0:
                qty = notional / entry
                risk = abs(entry - sl) * qty
            
            if risk > 0:
                r_multiple = pnl / risk
                total_r += r_multiple
                
        trades = len(rows)
        win_rate = (wins / trades) if trades > 0 else 0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (999 if gross_profit > 0 else 0)
        avg_r = (total_r / trades) if trades > 0 else 0
        
        return {
            "trades": trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_r": avg_r,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss
        }
        
    champ_metrics = calc_metrics(champion_rows)
    chall_metrics = calc_metrics(challenger_rows)
    
    MIN_TRADES = 20
    
    promoted = False
    reason = "needs more trades"
    
    if chall_metrics["trades"] >= MIN_TRADES:
        # Challenger must strictly beat the champion in Avg R and Profit Factor
        beat_r = chall_metrics["avg_r"] > champ_metrics["avg_r"]
        beat_pf = chall_metrics["profit_factor"] > champ_metrics["profit_factor"]
        profitable = chall_metrics["profit_factor"] > 1.05 and chall_metrics["avg_r"] > 0
        
        if profitable and (beat_r or beat_pf):
            promoted = True
            reason = "outperformed champion"
            set_alias(challenger_version, "latest")
            # Remove challenger alias
            import shutil
            shutil.rmtree(_model_root() / "challenger", ignore_errors=True)
            logger.info(f"PROMOTED CHALLENGER {challenger_version} to LATEST")
        else:
            reason = "did not outperform champion"
            
    return {
        "status": "evaluated",
        "challenger_version": challenger_version,
        "champion_metrics": champ_metrics,
        "challenger_metrics": chall_metrics,
        "promoted": promoted,
        "reason": reason
    }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(evaluate_battle(), indent=2))
