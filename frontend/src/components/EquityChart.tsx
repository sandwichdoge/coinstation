import { useEffect, useRef } from "react";
import {
  createChart,
  type IChartApi,
  type ISeriesApi,
  type UTCTimestamp,
} from "lightweight-charts";
import type { EquityPoint } from "../types";
import { baseChartOptions, COLORS } from "./chartTheme";

export function EquityChart({ data, height = 240 }: { data: EquityPoint[]; height?: number }) {
  const elRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const eqRef = useRef<ISeriesApi<"Line"> | null>(null);
  const bhRef = useRef<ISeriesApi<"Line"> | null>(null);

  useEffect(() => {
    if (!elRef.current) return;
    const chart = createChart(elRef.current, {
      ...baseChartOptions(),
      height,
      width: elRef.current.clientWidth,
    });
    eqRef.current = chart.addLineSeries({ color: COLORS.equity, lineWidth: 2 });
    bhRef.current = chart.addLineSeries({ color: COLORS.benchmark, lineWidth: 1, lineStyle: 2 });
    chartRef.current = chart;

    const ro = new ResizeObserver(() => {
      if (elRef.current) chart.applyOptions({ width: elRef.current.clientWidth });
    });
    ro.observe(elRef.current);
    return () => {
      ro.disconnect();
      chart.remove();
      chartRef.current = null;
    };
  }, [height]);

  useEffect(() => {
    if (!eqRef.current || !bhRef.current) return;
    eqRef.current.setData(data.map((d) => ({ time: d.time as UTCTimestamp, value: d.equity })));
    bhRef.current.setData(data.map((d) => ({ time: d.time as UTCTimestamp, value: d.buy_hold })));
    chartRef.current?.timeScale().fitContent();
  }, [data]);

  return <div ref={elRef} style={{ width: "100%" }} />;
}
