import { ColorType, CrosshairMode, type DeepPartial, type ChartOptions } from "lightweight-charts";

export const COLORS = {
  bg: "#0e0f14",
  text: "#c7c9d1",
  grid: "#1b1e29",
  border: "#2a2e3a",
  up: "#26a69a",
  down: "#ef5350",
  ema20: "#f5c542",
  ema50: "#4cc9f0",
  ema200: "#b388ff",
  bb: "#5c6370",
  macd: "#4cc9f0",
  signal: "#f5803e",
  equity: "#4cc9f0",
  benchmark: "#9aa0aa",
};

// Shared chart options. A fixed price-scale minimumWidth keeps stacked panes
// horizontally aligned (different value magnitudes would otherwise shift them).
export function baseChartOptions(): DeepPartial<ChartOptions> {
  return {
    layout: {
      background: { type: ColorType.Solid, color: COLORS.bg },
      textColor: COLORS.text,
      fontSize: 11,
    },
    grid: {
      vertLines: { color: COLORS.grid },
      horzLines: { color: COLORS.grid },
    },
    crosshair: { mode: CrosshairMode.Normal },
    rightPriceScale: { borderColor: COLORS.border, minimumWidth: 72 },
    timeScale: { borderColor: COLORS.border, timeVisible: true, secondsVisible: false },
  };
}
