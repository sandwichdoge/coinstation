import { useEffect, useRef } from "react";
import {
  createChart,
  LineStyle,
  type IChartApi,
  type ISeriesApi,
  type LogicalRange,
  type SeriesMarker,
  type Time,
  type UTCTimestamp,
} from "lightweight-charts";
import type { Candle, ChartMarker, HistPoint, Indicators, LinePoint } from "../types";
import { baseChartOptions, COLORS } from "./chartTheme";
import { fmtPrice } from "../util";

const toLine = (pts: LinePoint[]) => pts.map((p) => ({ time: p.time as UTCTimestamp, value: p.value }));
const toHist = (pts: HistPoint[]) =>
  pts.map((p) => ({ time: p.time as UTCTimestamp, value: p.value, color: p.color }));

interface Refs {
  charts: IChartApi[];
  candle: ISeriesApi<"Candlestick">;
  volume: ISeriesApi<"Histogram">;
  volMa: ISeriesApi<"Line">;
  ema20: ISeriesApi<"Line">;
  ema50: ISeriesApi<"Line">;
  ema200: ISeriesApi<"Line">;
  bbU: ISeriesApi<"Line">;
  bbM: ISeriesApi<"Line">;
  bbL: ISeriesApi<"Line">;
  rsi: ISeriesApi<"Line">;
  macd: ISeriesApi<"Line">;
  signal: ISeriesApi<"Line">;
  hist: ISeriesApi<"Histogram">;
}

interface Props {
  candles: Candle[];
  indicators?: Indicators | Record<string, never>;
  markers?: ChartMarker[];
  heights?: { price: number; volume: number; rsi: number; macd: number };
}

export function ChartStack({ candles, indicators, markers, heights }: Props) {
  const priceEl = useRef<HTMLDivElement>(null);
  const tipEl = useRef<HTMLDivElement>(null);
  const volEl = useRef<HTMLDivElement>(null);
  const rsiEl = useRef<HTMLDivElement>(null);
  const macdEl = useRef<HTMLDivElement>(null);
  const refs = useRef<Refs | null>(null);
  const lastKey = useRef<string>("");

  const H = heights ?? { price: 360, volume: 110, rsi: 120, macd: 150 };

  // ---- create charts once ----
  useEffect(() => {
    if (!priceEl.current || !volEl.current || !rsiEl.current || !macdEl.current) return;

    const price = createChart(priceEl.current, {
      ...baseChartOptions(),
      height: H.price,
      width: priceEl.current.clientWidth,
      timeScale: { ...baseChartOptions().timeScale, visible: false },
    });
    const vol = createChart(volEl.current, {
      ...baseChartOptions(),
      height: H.volume,
      width: volEl.current.clientWidth,
      timeScale: { ...baseChartOptions().timeScale, visible: false },
    });
    const rsi = createChart(rsiEl.current, {
      ...baseChartOptions(),
      height: H.rsi,
      width: rsiEl.current.clientWidth,
      timeScale: { ...baseChartOptions().timeScale, visible: false },
    });
    const macd = createChart(macdEl.current, {
      ...baseChartOptions(),
      height: H.macd,
      width: macdEl.current.clientWidth,
    });

    const candle = price.addCandlestickSeries({
      upColor: COLORS.up,
      downColor: COLORS.down,
      borderVisible: false,
      wickUpColor: COLORS.up,
      wickDownColor: COLORS.down,
    });
    // Trim the default top/bottom padding so candles fill the pane.
    price.priceScale("right").applyOptions({ scaleMargins: { top: 0.06, bottom: 0.08 } });
    const volume = vol.addHistogramSeries({
      priceFormat: { type: "volume" },
      priceLineVisible: false,
      lastValueVisible: false,
    });
    vol.priceScale("right").applyOptions({ scaleMargins: { top: 0.1, bottom: 0 } });

    const thinLine = (chart: IChartApi, color: string, style = LineStyle.Solid, width = 1) =>
      chart.addLineSeries({
        color,
        lineWidth: width as 1 | 2 | 3 | 4,
        lineStyle: style,
        priceLineVisible: false,
        lastValueVisible: false,
        crosshairMarkerVisible: false,
      });

    const ema20 = thinLine(price, COLORS.ema20);
    const ema50 = thinLine(price, COLORS.ema50);
    const ema200 = thinLine(price, COLORS.ema200, LineStyle.Solid, 2);
    const bbU = thinLine(price, COLORS.bb, LineStyle.Dashed);
    const bbM = thinLine(price, COLORS.bb, LineStyle.Dotted);
    const bbL = thinLine(price, COLORS.bb, LineStyle.Dashed);

    const volMa = thinLine(vol, COLORS.ema50);

    const rsiSeries = rsi.addLineSeries({ color: "#c792ea", lineWidth: 1, priceLineVisible: false });
    rsiSeries.createPriceLine({ price: 70, color: COLORS.down, lineStyle: LineStyle.Dashed, lineWidth: 1, title: "70" });
    rsiSeries.createPriceLine({ price: 30, color: COLORS.up, lineStyle: LineStyle.Dashed, lineWidth: 1, title: "30" });

    const hist = macd.addHistogramSeries({ priceLineVisible: false });
    const macdLine = thinLine(macd, COLORS.macd);
    const signalLine = thinLine(macd, COLORS.signal);
    macdLine.createPriceLine({ price: 0, color: COLORS.border, lineStyle: LineStyle.Dotted, lineWidth: 1, title: "" });

    const charts = [price, vol, rsi, macd];
    refs.current = {
      charts, candle, volume, volMa, ema20, ema50, ema200, bbU, bbM, bbL,
      rsi: rsiSeries, macd: macdLine, signal: signalLine, hist,
    };

    // ---- sync visible time range across panes ----
    let syncing = false;
    const link = (src: IChartApi) =>
      src.timeScale().subscribeVisibleLogicalRangeChange((range: LogicalRange | null) => {
        if (syncing || !range) return;
        syncing = true;
        for (const c of charts) if (c !== src) c.timeScale().setVisibleLogicalRange(range);
        syncing = false;
      });
    charts.forEach(link);

    // ---- keep widths in sync with the container ----
    const ro = new ResizeObserver(() => {
      const w = priceEl.current?.clientWidth ?? 0;
      if (w > 0) charts.forEach((c) => c.applyOptions({ width: w }));
    });
    if (priceEl.current) ro.observe(priceEl.current);

    // ---- OHLC tooltip following the cursor on the price pane ----
    price.subscribeCrosshairMove((param) => {
      const tip = tipEl.current;
      const host = priceEl.current;
      if (!tip || !host) return;
      const bar = param.time && param.point ? param.seriesData.get(candle) : undefined;
      if (!bar || !param.point) {
        tip.style.display = "none";
        return;
      }
      const o = bar as unknown as { open: number; high: number; low: number; close: number };
      const date = new Date((param.time as number) * 1000).toLocaleDateString(undefined, {
        year: "numeric",
        month: "short",
        day: "numeric",
      });
      const cls = o.close >= o.open ? COLORS.up : COLORS.down;
      tip.innerHTML =
        `<div class="cs-tip-date">${date}</div>` +
        `<div class="cs-tip-row"><span>O</span><b>${fmtPrice(o.open)}</b></div>` +
        `<div class="cs-tip-row"><span>C</span><b style="color:${cls}">${fmtPrice(o.close)}</b></div>` +
        `<div class="cs-tip-row"><span>H</span><b>${fmtPrice(o.high)}</b></div>` +
        `<div class="cs-tip-row"><span>L</span><b>${fmtPrice(o.low)}</b></div>`;
      tip.style.display = "block";

      // Keep the tooltip beside the cursor, flipping near the right/bottom edges.
      const pad = 12;
      const w = tip.offsetWidth;
      const h = tip.offsetHeight;
      let left = param.point.x + pad;
      if (left + w > host.clientWidth) left = param.point.x - w - pad;
      let top = param.point.y + pad;
      if (top + h > host.clientHeight) top = host.clientHeight - h - 2;
      tip.style.left = `${Math.max(2, left)}px`;
      tip.style.top = `${Math.max(2, top)}px`;
    });

    return () => {
      ro.disconnect();
      charts.forEach((c) => c.remove());
      refs.current = null;
      lastKey.current = "";
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [H.price, H.volume, H.rsi, H.macd]);

  // ---- push data on change ----
  useEffect(() => {
    const r = refs.current;
    if (!r) return;

    r.candle.setData(
      candles.map((c) => ({
        time: c.time as UTCTimestamp,
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close,
      })),
    );

    const ind = indicators as Indicators | undefined;
    const ov = ind?.overlays ?? {};
    r.ema20.setData(toLine(ov.ema20 ?? []));
    r.ema50.setData(toLine(ov.ema50 ?? []));
    r.ema200.setData(toLine(ov.ema200 ?? []));
    r.bbU.setData(toLine(ov.bb_upper ?? []));
    r.bbM.setData(toLine(ov.bb_mid ?? []));
    r.bbL.setData(toLine(ov.bb_lower ?? []));
    r.volume.setData(toHist(ind?.volume?.points ?? []));
    r.volMa.setData(toLine(ind?.volume?.ma ?? []));
    r.rsi.setData(toLine(ind?.rsi?.points ?? []));
    r.macd.setData(toLine(ind?.macd?.macd ?? []));
    r.signal.setData(toLine(ind?.macd?.signal ?? []));
    r.hist.setData(toHist(ind?.macd?.hist ?? []));

    const ms: SeriesMarker<Time>[] = (markers ?? []).map((m) => ({
      time: m.time as UTCTimestamp,
      position: m.position,
      color: m.color,
      shape: m.shape,
      text: m.text,
    }));
    r.candle.setMarkers(ms);

    // Reset zoom only when the dataset itself changes (symbol/interval/range),
    // not on incidental re-renders.
    const key = `${candles.length}:${candles[0]?.time ?? 0}:${candles[candles.length - 1]?.time ?? 0}`;
    if (key !== lastKey.current) {
      lastKey.current = key;
      r.charts.forEach((c) => c.timeScale().fitContent());
    }

    // Equalize right-scale widths so the stacked panes line up vertically.
    requestAnimationFrame(() => {
      const cs = refs.current?.charts;
      if (!cs) return;
      const max = Math.max(...cs.map((c) => c.priceScale("right").width()));
      if (Number.isFinite(max) && max > 0) {
        cs.forEach((c) => c.priceScale("right").applyOptions({ minimumWidth: max }));
      }
    });
  }, [candles, indicators, markers]);

  return (
    <div className="chart-stack">
      <div className="chart-pane price-pane">
        <div ref={priceEl} />
        <div className="cs-tooltip" ref={tipEl} />
      </div>
      <div className="chart-pane-label">Volume (MA 20)</div>
      <div className="chart-pane" ref={volEl} />
      <div className="chart-pane-label">RSI ({(indicators as Indicators)?.rsi?.period ?? 14})</div>
      <div className="chart-pane" ref={rsiEl} />
      <div className="chart-pane-label">MACD (12, 26, 9)</div>
      <div className="chart-pane" ref={macdEl} />
    </div>
  );
}
