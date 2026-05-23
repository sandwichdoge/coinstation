import { useCallback, useEffect, useState } from "react";
import { api } from "./api/client";
import type { AnalysisResult, Coin, Health, Indicators, KlinesResponse, NewsItem, Snapshot } from "./types";
import { Controls } from "./components/Controls";
import { ChartStack } from "./components/ChartStack";
import { NewsPanel } from "./components/NewsPanel";
import { AnalysisPanel } from "./components/AnalysisPanel";
import { BacktestPanel } from "./components/BacktestPanel";
import { COLORS } from "./components/chartTheme";
import { daysAgoISO, errMsg, fmtPct, fmtPrice } from "./util";

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [coins, setCoins] = useState<Coin[]>([]);
  const [intervals, setIntervals] = useState<string[]>([]);

  const [symbol, setSymbol] = useState("BTCUSDT");
  const [interval, setIntervalSel] = useState("1h");
  const [mode, setMode] = useState<"live" | "range">("live");
  const [start, setStart] = useState(daysAgoISO(180));
  const [end, setEnd] = useState("");

  const [klines, setKlines] = useState<KlinesResponse | null>(null);
  const [chartLoading, setChartLoading] = useState(false);
  const [chartError, setChartError] = useState<string | null>(null);

  const [news, setNews] = useState<NewsItem[]>([]);
  const [newsLoading, setNewsLoading] = useState(false);

  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [analysisError, setAnalysisError] = useState<string | null>(null);

  const asOf = mode === "range" && end ? end : undefined;

  useEffect(() => {
    api.health().then(setHealth).catch(() => {});
    api.coins().then((r) => setCoins(r.coins)).catch(() => {});
    api.intervals().then((r) => setIntervals(r.intervals)).catch(() => {});
  }, []);

  const fetchChart = useCallback(async () => {
    setChartLoading(true);
    setChartError(null);
    try {
      const r =
        mode === "live"
          ? await api.klines({ symbol, interval, limit: 500, with_indicators: true })
          : await api.klines({ symbol, interval, start, end: end || undefined, warmup: true, with_indicators: true });
      setKlines(r);
    } catch (e) {
      setChartError(errMsg(e));
      setKlines(null);
    } finally {
      setChartLoading(false);
    }
  }, [symbol, interval, mode, start, end]);

  const fetchNews = useCallback(async () => {
    setNewsLoading(true);
    try {
      const r = await api.news({ symbol, before: asOf, limit: 20 });
      setNews(r.items);
    } catch {
      setNews([]);
    } finally {
      setNewsLoading(false);
    }
  }, [symbol, asOf]);

  // Auto-load on coin / timeframe / mode changes. Date edits apply on Refresh.
  useEffect(() => {
    fetchChart();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [symbol, interval, mode]);
  useEffect(() => {
    fetchNews();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [symbol, asOf]);

  async function runAnalyze() {
    setAnalysisLoading(true);
    setAnalysisError(null);
    try {
      setAnalysis(await api.analyze({ symbol, interval, as_of: asOf }));
    } catch (e) {
      setAnalysisError(errMsg(e));
      setAnalysis(null);
    } finally {
      setAnalysisLoading(false);
    }
  }

  const candles = klines?.candles ?? [];
  const last = candles[candles.length - 1];
  const prev = candles[candles.length - 2];
  const snap = (klines?.snapshot ?? {}) as Snapshot;
  const indicators = klines?.indicators as Indicators | undefined;
  // Change vs the previous bar's open (e.g. yesterday's open on a 1d chart),
  // rather than a whole-window delta.
  const chg =
    last && prev ? (last.close / prev.open - 1) * 100 : undefined;

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <h1>
            Coin<span className="dot">Station</span>
          </h1>
          <span className="tag-line">price action · indicators · news · analysis · backtest</span>
        </div>
        <span className={`badge ${health ? "on" : "off"}`}>
          {health ? "Rule-based engine" : "connecting…"}
        </span>
      </header>

      <Controls
        coins={coins}
        intervals={intervals}
        symbol={symbol}
        onSymbol={setSymbol}
        interval={interval}
        onInterval={setIntervalSel}
        mode={mode}
        onMode={setMode}
        start={start}
        onStart={setStart}
        end={end}
        onEnd={setEnd}
        onRefresh={() => {
          fetchChart();
          fetchNews();
        }}
        loading={chartLoading}
      />

      <div className="grid-main">
        <div className="chart-col">
          <div className="panel">
            <div className="price-head">
              <span className="sym">{symbol}</span>
              <span className="px">{fmtPrice(last?.close)}</span>
              {chg != null && <span className={`chg ${chg >= 0 ? "up" : "down"}`}>{fmtPct(chg)}</span>}
              <span className="muted">
                {interval} ·{" "}
                {mode === "live" ? `last ${candles.length} bars` : `${start} → ${end || "now"}`}
              </span>
            </div>

            {chartError && <div className="error">{chartError}</div>}
            {chartLoading && candles.length === 0 && <div className="spinner">Loading chart…</div>}

            <ChartStack candles={candles} indicators={indicators} />

            <div className="legend">
              <span style={{ color: COLORS.ema20 }}>EMA20</span>
              <span style={{ color: COLORS.ema50 }}>EMA50</span>
              <span style={{ color: COLORS.ema200 }}>EMA200</span>
              <span style={{ color: COLORS.bb }}>Bollinger(20,2)</span>
            </div>

            {snap.signals && snap.signals.length > 0 && (
              <div className="row" style={{ marginTop: 8 }}>
                {snap.signals.map((s) => (
                  <span className="tag" key={s}>
                    {s}
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>

        <div className="side-col">
          <AnalysisPanel
            result={analysis}
            loading={analysisLoading}
            error={analysisError}
            onAnalyze={runAnalyze}
          />
          <NewsPanel items={news} loading={newsLoading} asOf={asOf} onRefresh={fetchNews} />
        </div>
      </div>

      <BacktestPanel symbol={symbol} />
    </div>
  );
}
