// Shapes mirror the FastAPI responses.

export interface Coin {
  symbol: string;
  base: string;
  name: string;
}

export interface Candle {
  time: number; // UNIX seconds (UTC)
  open: number;
  high: number;
  low: number;
  close: number;
}

export interface LinePoint {
  time: number;
  value: number;
}

export interface HistPoint {
  time: number;
  value: number;
  color: string;
}

export interface Indicators {
  overlays: Record<string, LinePoint[]>; // ema20/50/200, bb_upper/mid/lower
  rsi: { period: number; points: LinePoint[] };
  macd: { macd: LinePoint[]; signal: LinePoint[]; hist: HistPoint[] };
  volume: { points: HistPoint[]; ma: LinePoint[] };
}

export interface Snapshot {
  time?: number;
  price?: number | null;
  rsi?: number | null;
  macd?: number | null;
  macd_signal?: number | null;
  macd_hist?: number | null;
  ema20?: number | null;
  ema50?: number | null;
  ema200?: number | null;
  bb_upper?: number | null;
  bb_lower?: number | null;
  signals?: string[];
  window_change_pct?: number;
}

export interface KlinesResponse {
  symbol: string;
  interval: string;
  candles: Candle[];
  indicators: Indicators | Record<string, never>;
  snapshot: Snapshot | Record<string, never>;
}

export interface NewsItem {
  id: number;
  source: string;
  title: string;
  summary: string;
  link: string;
  symbols: string[];
  published_at: string;
}

export type Action = "strong_buy" | "buy" | "hold" | "sell" | "strong_sell";

export interface AnalysisResult {
  symbol: string;
  as_of: string;
  interval: string;
  action: Action;
  confidence: number;
  horizon: string;
  summary: string;
  rationale: string;
  key_factors: string[];
  risks: string[];
  technical_snapshot: Snapshot;
  news_considered: number;
  headlines: { title: string; source: string; published_at: string; link?: string }[];
  source: string;
  model: string | null;
}

export interface Trade {
  time: number;
  side: "buy" | "sell";
  price: number;
  units: number;
  cash?: number;
}

export interface ChartMarker {
  time: number;
  position: "aboveBar" | "belowBar";
  color: string;
  shape: "arrowUp" | "arrowDown";
  text: string;
}

export interface EquityPoint {
  time: number;
  equity: number;
  buy_hold: number;
}

export interface BacktestMetrics {
  initial_capital: number;
  final_equity: number;
  total_return_pct: number;
  buy_hold_return_pct: number;
  alpha_pct: number;
  max_drawdown_pct: number;
  sharpe: number;
  num_trades: number;
  win_rate_pct: number | null;
  exposure_pct: number | null;
}

export interface BacktestResult {
  symbol: string;
  interval: string;
  strategy: string;
  start: string;
  end: string;
  bars: number;
  metrics: BacktestMetrics;
  equity_curve: EquityPoint[];
  trades: Trade[];
  markers: ChartMarker[];
  candles: Candle[];
  overlays: Record<string, LinePoint[]>;
  volume: { points: HistPoint[]; ma: LinePoint[] };
  rsi: { period: number; points: LinePoint[] };
  macd: { macd: LinePoint[]; signal: LinePoint[]; hist: HistPoint[] };
}

export interface Health {
  status: string;
  ai_enabled: boolean;
  model: string | null;
}
