
import React, { useState } from "react";

export default function Projection({ apiBase }) {
  const [texts, setTexts] = useState("");
  const [method, setMethod] = useState("umap");
  const [coords, setCoords] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const project = async () => {
    const list = texts.split("\n").map(s => s.trim()).filter(Boolean);
    if (!list.length) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${apiBase}/api/ml/project`, {
        method: "POST",
        headers: { "Content-Type":"application/json" },
        body: JSON.stringify({ texts: list, method }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.detail || "Projection failed");
      setCoords(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ marginTop: 16 }}>
      <h3>Embedding projection ({method.toUpperCase()})</h3>
      <p>Paste 5–50 lines of text. We'll embed with a sentence transformer and project to 2D.</p>
      <textarea value={texts} onChange={(e)=>setTexts(e.target.value)} rows={6} style={{width:"100%"}} placeholder={"Line 1\nLine 2\n…"} />
      <div style={{ display:"flex", gap:8, alignItems:"center", marginTop:8 }}>
        <label>Method:</label>
        <select value={method} onChange={(e)=>setMethod(e.target.value)}>
          <option value="umap">UMAP</option>
          <option value="tsne">t‑SNE</option>
        </select>
        <button onClick={project} disabled={loading || !texts.trim()}>Project</button>
      </div>
      {error && <p style={{ color:"crimson" }}>{error}</p>}
      {loading && <p>Computing…</p>}
      {coords && (
        <svg width="100%" height="320" viewBox="-1 -1 2 2" preserveAspectRatio="xMidYMid meet" style={{ background:"#fafafa", border:"1px solid #eee", marginTop: 12 }}>
          {coords.x.map((x, i) => {
            const minX = Math.min(...coords.x), maxX = Math.max(...coords.x);
            const minY = Math.min(...coords.y), maxY = Math.max(...coords.y);
            const cx = (x - minX) / (maxX - minX + 1e-6) * 2 - 1;
            const cy = (coords.y[i] - minY) / (maxY - minY + 1e-6) * 2 - 1;
            return <circle key={i} cx={cx} cy={-cy} r={0.02} />
          })}
        </svg>
      )}
    </div>
  );
}
