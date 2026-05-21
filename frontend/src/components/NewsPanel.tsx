import type { NewsItem } from "../types";
import { relTime } from "../util";

interface Props {
  items: NewsItem[];
  loading: boolean;
  asOf?: string;
  onRefresh: () => void;
}

export function NewsPanel({ items, loading, asOf, onRefresh }: Props) {
  return (
    <div className="panel">
      <div className="panel-title">
        <span>
          News {asOf && <span className="sub">as of {new Date(asOf).toLocaleDateString()}</span>}
        </span>
        <button className="btn ghost" style={{ padding: "4px 10px" }} onClick={onRefresh} disabled={loading}>
          {loading ? "…" : "↻"}
        </button>
      </div>
      {!loading && items.length === 0 && (
        <div className="empty">
          No news in the archive for this coin/date. The backend ingests RSS over time — historical
          depth grows as it runs.
        </div>
      )}
      <div className="news-list">
        {items.map((n) => (
          <div className="news-item" key={n.id}>
            <a className="title" href={n.link} target="_blank" rel="noreferrer">
              {n.title}
            </a>
            <div className="meta">
              <span>{n.source}</span>
              <span>·</span>
              <span>{relTime(n.published_at)}</span>
              {n.symbols.map((s) => (
                <span className="tag" key={s}>
                  {s}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
