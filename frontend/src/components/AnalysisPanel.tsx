import type { AnalysisResult } from "../types";
import { actionLabel, fmtNum, fmtPrice } from "../util";

interface Props {
  result: AnalysisResult | null;
  loading: boolean;
  error: string | null;
  aiEnabled: boolean | null;
  onAnalyze: () => void;
}

export function AnalysisPanel({ result, loading, error, aiEnabled, onAnalyze }: Props) {
  const snap = result?.technical_snapshot;
  return (
    <div className="panel analysis">
      <div className="panel-title">
        <span>
          AI Analysis{" "}
          <span className="sub">{aiEnabled ? "OpenAI" : "rule-based (no API key)"}</span>
        </span>
        <button className="btn" style={{ padding: "5px 12px" }} onClick={onAnalyze} disabled={loading}>
          {loading ? "Analyzing…" : "Analyze"}
        </button>
      </div>

      {error && <div className="error">{error}</div>}

      {!result && !error && (
        <div className="empty">
          Combines the technical snapshot with recent news (as of the selected time) into an
          action and confidence score.
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
              {(snap.support != null || snap.resistance != null) && (
                <>
                  <div className="k">Support / Resist.</div>
                  <div>
                    {fmtPrice(snap.support)} / {fmtPrice(snap.resistance)}
                  </div>
                </>
              )}
            </div>
          )}

          <div className="foot">
            {result.source === "openai" ? `Model: ${result.model}` : "Rule-based engine"} · considered{" "}
            {result.news_considered} headline(s) · as of {new Date(result.as_of).toLocaleString()}
          </div>
        </>
      )}
    </div>
  );
}
