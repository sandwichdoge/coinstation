import type { Coupling, DecoupleResult } from "../types";
import { fmtNum, fmtPct } from "../util";

interface Props {
  result: DecoupleResult | null;
  loading: boolean;
  error: string | null;
}

const COUPLING_LABEL: Record<Coupling, string> = {
  coupled: "Coupled",
  loosening: "Loosening",
  decoupled: "Decoupled",
  inverse: "Inverse",
};

export function DecouplePanel({ result, loading, error }: Props) {
  const m = result?.meter;
  const score = m?.decouple_score ?? 0;

  return (
    <div className="panel decouple">
      <div className="panel-title">
        <span>
          Decouple meter <span className="sub">vs BTC sentiment</span>
        </span>
        {result && !result.is_reference && m?.available && (
          <span className="muted">{m.samples} bars · {result.interval}</span>
        )}
      </div>

      {error && <div className="error">{error}</div>}
      {loading && !result && <div className="empty">Measuring decoupling…</div>}

      {result?.is_reference && (
        <div className="empty">
          {result.symbol} is the reference. The meter scores how far <em>other</em> coins drift
          from BTC, the market's king — pick another coin to read it.
        </div>
      )}

      {result && !result.is_reference && m && !m.available && (
        <div className="empty">{m.reason ?? "Not enough aligned data to compare with BTC."}</div>
      )}

      {result && !result.is_reference && m?.available && (
        <>
          <div className="dc-gauge-row">
            <div className={`dc-score dc-${m.coupling}`}>{fmtNum(score, 0)}</div>
            <div className="dc-gauge-body">
              <div className="dc-gauge-head">
                <span className={`dc-badge dc-${m.coupling}`}>{COUPLING_LABEL[m.coupling!]}</span>
                {m.direction && m.direction !== "neutral" && (
                  <span className={`dc-dir dc-dir-${m.direction}`}>
                    {m.direction === "strength" ? "▲ relative strength" : "▼ relative weakness"}
                  </span>
                )}
              </div>
              <div className="dc-bar">
                <div className={`dc-fill dc-${m.coupling}`} style={{ width: `${score}%` }} />
              </div>
              <div className="dc-scale">
                <span>tracks BTC</span>
                <span>trades on its own</span>
              </div>
            </div>
          </div>

          {m.summary && <p className="dc-summary">{m.summary}</p>}

          <div className="snap-grid">
            <div className="k">Correlation</div>
            <div>{fmtNum(m.correlation, 2)}</div>
            <div className="k">Beta vs BTC</div>
            <div>{m.beta == null ? "—" : `${fmtNum(m.beta, 2)}×`}</div>
            <div className="k">{result.symbol.replace("USDT", "")} move</div>
            <div className={(m.coin_change_pct ?? 0) >= 0 ? "up" : "down"}>{fmtPct(m.coin_change_pct, 1)}</div>
            <div className="k">BTC move</div>
            <div className={(m.btc_change_pct ?? 0) >= 0 ? "up" : "down"}>{fmtPct(m.btc_change_pct, 1)}</div>
            <div className="k">Rel. strength</div>
            <div className={(m.relative_strength_pct ?? 0) >= 0 ? "up" : "down"}>
              {fmtPct(m.relative_strength_pct, 1)}
            </div>
            <div className="k">Alpha (β-adj.)</div>
            <div className={(m.alpha_pct ?? 0) >= 0 ? "up" : "down"}>
              {m.alpha_pct == null ? "—" : fmtPct(m.alpha_pct, 1)}
            </div>
          </div>

          <div className="foot">
            Decouple score = 1 − R² over {m.samples} candles · BTC = global sentiment proxy
          </div>
        </>
      )}
    </div>
  );
}
