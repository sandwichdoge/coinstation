export function fmtPrice(v?: number | null): string {
  if (v == null || Number.isNaN(v)) return "—";
  const abs = Math.abs(v);
  const digits = abs >= 1 ? 2 : abs >= 0.01 ? 4 : 8;
  return v.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export function fmtNum(v?: number | null, d = 2): string {
  if (v == null || Number.isNaN(v)) return "—";
  return v.toLocaleString(undefined, { maximumFractionDigits: d });
}

export function fmtPct(v?: number | null, d = 2): string {
  if (v == null || Number.isNaN(v)) return "—";
  return `${v >= 0 ? "+" : ""}${v.toFixed(d)}%`;
}

export function fmtMoney(v?: number | null): string {
  if (v == null || Number.isNaN(v)) return "—";
  return `$${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
}

export function relTime(iso: string): string {
  const t = new Date(iso).getTime();
  if (Number.isNaN(t)) return "";
  const s = Math.floor((Date.now() - t) / 1000);
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.floor(h / 24);
  if (d < 30) return `${d}d ago`;
  return new Date(iso).toLocaleDateString();
}

export const actionLabel = (a: string) => a.replace(/_/g, " ").toUpperCase();

// Default ISO date (YYYY-MM-DD) `days` before now, for date pickers.
export function daysAgoISO(days: number): string {
  const d = new Date(Date.now() - days * 86400000);
  return d.toISOString().slice(0, 10);
}

export const errMsg = (e: unknown) => (e instanceof Error ? e.message : String(e));
