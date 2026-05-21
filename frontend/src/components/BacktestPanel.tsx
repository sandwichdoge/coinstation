import { useState } from "react";
import { api, type BacktestBody } from "../api/client";
import type { BacktestResult, Indicators } from "../types";
import { ChartStack } from "./ChartStack";
import { EquityChart } from "./EquityChart";
import { daysAgoISO, errMsg, fmtMoney, fmtNum, fmtPct } from "../util";

const STRATEGIES = [
  { id: "macd", label: "MACD crossover" },
  { id: "rsi", label: "RSI bands" },
  { id: "ema_cross", label: "EMA 50/200 cross" },
  { id: "ai", label: "AI (news + technicals)" },
  { id: "buy_hold", label: "Buy & Hold" },
];
const BT_INTERVALS = ["1h", "2h", "4h", "6h", "12h", "1d", "3d", "1w"];

function Metric({ label, value, tone }: { label: string; value: string; tone?: "up" | "down" }) {
  return (
    <div className="metric">
      <div className="label">{label}</div>
      <div className={`value ${tone ?? ""}`}>{value}</div>
    </div>
  );
}

const tone = (v: number): "up" | "down" => (v >= 0 ? "up" : "down");

export function BacktestPanel({ symbol, aiEnabled }: { symbol: string; aiEnabled: boolean | null }) {
  const [strategy, setStrategy] = useState("macd");
  const [interval, setInterval] = useState("1d");
  const [start, setStart] = useState(daysAgoISO(365 * 2));
  const [end, setEnd] = useState("");
  const [capital, setCapital] = useState(10000);
  const [fee, setFee] = useState(0.1);
  const [rsiBuy, setRsiBuy] = useState(30);
  const [rsiSell, setRsiSell] = useState(70);
  const [aiEvery, setAiEvery] = useState(14);
  const [aiConf, setAiConf] = useState(60);

  const [result, setResult] = useState<BacktestResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    setLoading(true);
    setError(null);
    try {
      const body: BacktestBody = {
        symbol,
        interval,
        start,
        end: end || undefined,
        strategy,
        initial_capital: capital,
        fee_pct: fee,
        rsi_buy: rsiBuy,
        rsi_sell: rsiSell,
        ai_rebalance_every: aiEvery,
        ai_confidence_threshold: aiConf,
      };
      setResult(await api.backtest(body));
    } catch (e) {
      setError(errMsg(e));
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  const indicators: Indicators | undefined = result
    ? { overlays: result.overlays, rsi: result.rsi, macd: result.macd, volume: result.volume }
    : undefined;
  const m = result?.metrics;

  return (
    <div className="panel section-gap">
      <div className="panel-title">
        <span>
          Backtest <span className="sub">{symbol} · date-aware, news included</span>
        </span>
      </div>

      <div className="bt-form">
        <div className="control-group">
          <label>Strategy</label>
          <select value={strategy} onChange={(e) => setStrategy(e.target.value)}>
            {STRATEGIES.map((s) => (
              <option key={s.id} value={s.id}>
                {s.label}
              </option>
            ))}
          </select>
        </div>
        <div className="control-group">
          <label>Interval</label>
          <select value={interval} onChange={(e) => setInterval(e.target.value)}>
            {BT_INTERVALS.map((iv) => (
              <option key={iv} value={iv}>
                {iv}
              </option>
            ))}
          </select>
        </div>
        <div className="control-group">
          <label>From</label>
          <input type="date" value={start} onChange={(e) => setStart(e.target.value)} />
        </div>
        <div className="control-group">
          <label>To (blank = now)</label>
          <input type="date" value={end} onChange={(e) => setEnd(e.target.value)} />
        </div>
        <div className="control-group">
          <label>Capital ($)</label>
          <input type="number" value={capital} min={100} step={100} onChange={(e) => setCapital(+e.target.value)} style={{ width: 100 }} />
        </div>
        <div className="control-group">
          <label>Fee (%)</label>
          <input type="number" value={fee} min={0} step={0.05} onChange={(e) => setFee(+e.target.value)} style={{ width: 70 }} />
        </div>

        {strategy === "rsi" && (
          <>
            <div className="control-group">
              <label>RSI buy ≤</label>
              <input type="number" value={rsiBuy} onChange={(e) => setRsiBuy(+e.target.value)} style={{ width: 70 }} />
            </div>
            <div className="control-group">
              <label>RSI sell ≥</label>
              <input type="number" value={rsiSell} onChange={(e) => setRsiSell(+e.target.value)} style={{ width: 70 }} />
            </div>
          </>
        )}
        {strategy === "ai" && (
          <>
            <div className="control-group">
              <label>Eval every (bars)</label>
              <input type="number" value={aiEvery} min={1} onChange={(e) => setAiEvery(+e.target.value)} style={{ width: 90 }} />
            </div>
            <div className="control-group">
              <label>Min confidence</label>
              <input type="number" value={aiConf} min={0} max={100} onChange={(e) => setAiConf(+e.target.value)} style={{ width: 90 }} />
            </div>
          </>
        )}

        <div className="control-group">
          <label>&nbsp;</label>
          <button className="btn" onClick={run} disabled={loading}>
            {loading ? "Running…" : "Run backtest"}
          </button>
        </div>
      </div>

      {strategy === "ai" && (
        <div className="empty" style={{ marginTop: -4 }}>
          {aiEnabled
            ? "AI strategy calls OpenAI at each evaluation bar with news as of that date — this can be slow/costly over long ranges."
            : "AI strategy runs with the rule-based engine (no OPENAI_API_KEY set), so it stays fast and free."}
        </div>
      )}

      {error && <div className="error">{error}</div>}

      {m && result && (
        <>
          <div className="metrics-grid">
            <Metric label="Total return" value={fmtPct(m.total_return_pct)} tone={tone(m.total_return_pct)} />
            <Metric label="Buy & hold" value={fmtPct(m.buy_hold_return_pct)} tone={tone(m.buy_hold_return_pct)} />
            <Metric label="Alpha vs B&H" value={fmtPct(m.alpha_pct)} tone={tone(m.alpha_pct)} />
            <Metric label="Max drawdown" value={fmtPct(m.max_drawdown_pct)} tone="down" />
            <Metric label="Sharpe" value={fmtNum(m.sharpe, 2)} />
            <Metric label="Trades" value={String(m.num_trades)} />
            <Metric label="Win rate" value={m.win_rate_pct == null ? "—" : `${fmtNum(m.win_rate_pct, 0)}%`} />
            <Metric label="Final equity" value={fmtMoney(m.final_equity)} />
          </div>

          <div className="panel-title">
            <span className="sub">
              Equity curve — strategy (blue) vs buy & hold (grey) · {result.bars} bars · exposure{" "}
              {fmtNum(m.exposure_pct, 0)}%
            </span>
          </div>
          <EquityChart data={result.equity_curve} height={220} />

          <div className="panel-title" style={{ marginTop: 16 }}>
            <span className="sub">Price with entries/exits</span>
          </div>
          <ChartStack
            candles={result.candles}
            indicators={indicators}
            markers={result.markers}
            heights={{ price: 320, rsi: 110, macd: 130 }}
          />

          {result.trades.length > 0 && (
            <div className="trades-scroll">
              <table className="trades-table">
                <thead>
                  <tr>
                    <th>Date</th>
                    <th>Side</th>
                    <th>Price</th>
                    <th>Units</th>
                  </tr>
                </thead>
                <tbody>
                  {result.trades.map((t, i) => (
                    <tr key={i}>
                      <td>{new Date(t.time * 1000).toLocaleDateString()}</td>
                      <td className={t.side === "buy" ? "up" : "down"}>{t.side.toUpperCase()}</td>
                      <td>{fmtNum(t.price, 2)}</td>
                      <td>{fmtNum(t.units, 4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </div>
  );
}
