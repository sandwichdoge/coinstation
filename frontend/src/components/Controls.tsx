import type { Coin } from "../types";

interface Props {
  coins: Coin[];
  intervals: string[];
  symbol: string;
  onSymbol: (s: string) => void;
  interval: string;
  onInterval: (s: string) => void;
  mode: "live" | "range";
  onMode: (m: "live" | "range") => void;
  start: string;
  onStart: (s: string) => void;
  end: string;
  onEnd: (s: string) => void;
  onRefresh: () => void;
  loading: boolean;
}

export function Controls(p: Props) {
  return (
    <div className="controls">
      <div className="control-group">
        <label>Coin</label>
        <select value={p.symbol} onChange={(e) => p.onSymbol(e.target.value)}>
          {p.coins.map((c) => (
            <option key={c.symbol} value={c.symbol}>
              {c.name} ({c.base})
            </option>
          ))}
        </select>
      </div>

      <div className="control-group">
        <label>Timeframe</label>
        <div className="seg">
          {p.intervals.map((iv) => (
            <button
              key={iv}
              className={`seg-btn ${iv === p.interval ? "active" : ""}`}
              onClick={() => p.onInterval(iv)}
            >
              {iv}
            </button>
          ))}
        </div>
      </div>

      <div className="control-group">
        <label>Mode</label>
        <div className="seg">
          <button className={`seg-btn ${p.mode === "live" ? "active" : ""}`} onClick={() => p.onMode("live")}>
            Live
          </button>
          <button className={`seg-btn ${p.mode === "range" ? "active" : ""}`} onClick={() => p.onMode("range")}>
            Historical
          </button>
        </div>
      </div>

      {p.mode === "range" && (
        <>
          <div className="control-group">
            <label>From</label>
            <input type="date" value={p.start} onChange={(e) => p.onStart(e.target.value)} />
          </div>
          <div className="control-group">
            <label>To (blank = now)</label>
            <input type="date" value={p.end} onChange={(e) => p.onEnd(e.target.value)} />
          </div>
        </>
      )}

      <div className="control-group">
        <label>&nbsp;</label>
        <button className="btn" onClick={p.onRefresh} disabled={p.loading}>
          {p.loading ? "Loading…" : "Refresh"}
        </button>
      </div>
    </div>
  );
}
