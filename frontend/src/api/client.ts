import type {
  AnalysisResult,
  BacktestResult,
  Coin,
  Health,
  KlinesResponse,
  NewsItem,
} from "../types";

const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "/api";

function qs(params: Record<string, string | number | boolean | undefined | null>): string {
  const sp = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== "") sp.set(k, String(v));
  }
  const s = sp.toString();
  return s ? `?${s}` : "";
}

async function http<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "content-type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

export interface KlinesParams {
  symbol: string;
  interval: string;
  start?: string;
  end?: string;
  limit?: number;
  with_indicators?: boolean;
  warmup?: boolean;
}

export interface AnalyzeBody {
  symbol: string;
  interval: string;
  as_of?: string;
  lookback?: number;
  news_limit?: number;
}

export interface BacktestBody {
  symbol: string;
  interval: string;
  start: string;
  end?: string;
  strategy: string;
  initial_capital: number;
  fee_pct: number;
  rsi_buy?: number;
  rsi_sell?: number;
  rules_rebalance_every?: number;
  rules_confidence_threshold?: number;
}

export const api = {
  health: () => http<Health>("/health"),
  coins: () => http<{ coins: Coin[] }>("/coins"),
  intervals: () => http<{ intervals: string[] }>("/intervals"),
  klines: (p: KlinesParams) => http<KlinesResponse>(`/klines${qs({ ...p })}`),
  news: (p: { symbol?: string; before?: string; limit?: number }) =>
    http<{ count: number; items: NewsItem[] }>(`/news${qs({ ...p })}`),
  ingestNews: () => http<{ added: number; fetched: number }>("/news/ingest", { method: "POST" }),
  analyze: (body: AnalyzeBody) =>
    http<AnalysisResult>("/analyze", { method: "POST", body: JSON.stringify(body) }),
  backtest: (body: BacktestBody) =>
    http<BacktestResult>("/backtest", { method: "POST", body: JSON.stringify(body) }),
};
