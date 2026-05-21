import type { AnalysisResult } from "../types";
import { actionLabel, fmtNum, fmtPrice } from "../util";

interface Props {
  result: AnalysisResult | null;
  loading: boolean;
  error: string | null;
  onAnalyze: () => void;
}

export function AnalysisPanel({ result, loading, error, onAnalyze }: Props) {
  const snap = result?.technical_snapshot;
  return (
    <div className="panel analysis">
      <div className="panel-title">
        <span>
          Analysis{" "}
          <span className="sub">rule-based</span>
        </span>
        <button className="btn" style={{ padding: "5px 12px" }} onClick={onAnalyze} disabled={loading}>
          {loading ? "Analyzing…" : "Analyze"}
        </button>
      </div>

      {error && <div className="error">{error}</div>}

      {!result && !error && (
        <div className="empty">
          Scores the technical snapshot (as of the selected time) into an action and confidence.
          Recent headlines are shown for context but don't affect the score.
        </div>
      )}

      {result && (
        <>
          <div className="row">
            <span className={`action-badge action-${result.action}`}>{actionLabel(result.action)}</span>
            <span className="muted">{result.horizon}</span>
          </div>

          <div className="conf-wrap">
            <div className="row" style={{ justifyContent: "space-between" }}>
              <span className="muted">Confidence</span>
              <strong>{fmtNum(result.confidence, 0)}%</strong>
            </div>
            <div className="conf-bar">
              <div className="conf-fill" style={{ width: `${result.confidence}%` }} />
            </div>
          </div>

          <p>
            <strong>{result.summary}</strong>
          </p>
          {result.rationale && <p>{result.rationale}</p>}

          {result.key_factors.length > 0 && (
            <ul className="factors">
              {result.key_factors.map((f, i) => (
                <li key={i}>{f}</li>
              ))}
            </ul>
          )}
          {result.risks.length > 0 && (
            <ul className="factors risk">
              {result.risks.map((r, i) => (
                <li key={i}>{r}</li>
              ))}
            </ul>
          )}

          {snap && (snap.support != null || snap.resistance != null) && (
            <div className="levels">
              <div className="levels-title">Key levels</div>
              {snap.resistance != null && (
                <div className="level res">
                  <div className="level-head">
                    <span className="level-tag">Resistance</span>
                    <span className="level-price">{fmtPrice(snap.resistance)}</span>
                    <span className="level-meta">
                      {snap.resistance_dist_pct != null && `${fmtNum(snap.resistance_dist_pct, 1)}% above`}
                      {snap.resistance_touches != null && ` · rejected ${snap.resistance_touches}×`}
                    </span>
                  </div>
                  <div className="level-next">
                    {snap.resistance_next != null
                      ? `If breached → next ~${fmtPrice(snap.resistance_next)}`
                      : "If breached → open air above"}
                  </div>
                </div>
              )}
              {snap.support != null && (
                <div className="level sup">
                  <div className="level-head">
                    <span className="level-tag">Support</span>
                    <span className="level-price">{fmtPrice(snap.support)}</span>
                    <span className="level-meta">
                      {snap.support_dist_pct != null && `${fmtNum(snap.support_dist_pct, 1)}% below`}
                      {snap.support_touches != null && ` · held ${snap.support_touches}×`}
                    </span>
                  </div>
                  <div className="level-next">
                    {snap.support_next != null
                      ? `If breached → next ~${fmtPrice(snap.support_next)}`
                      : "If breached → no clear level beneath"}
                  </div>
                </div>
              )}
            </div>
          )}

          {snap && (
            <div className="snap-grid">
              <div className="k">Price</div>
              <div>{fmtPrice(snap.price)}</div>
              <div className="k">RSI</div>
              <div>{fmtNum(snap.rsi, 1)}</div>
              <div className="k">MACD hist</div>
              <div>{fmtNum(snap.macd_hist, 2)}</div>
              <div className="k">EMA50 / 200</div>
              <div>
                {fmtPrice(snap.ema50)} / {fmtPrice(snap.ema200)}
              </div>
            </div>
          )}

          <div className="foot">
            Rule-based engine · considered {result.news_considered} headline(s) · as of{" "}
            {new Date(result.as_of).toLocaleString()}
          </div>
        </>
      )}
    </div>
  );
}
