
import React, { useState } from "react";

export default function Explain({ apiBase, selectedModel }) {
  const [text, setText] = useState("");
  const [method, setMethod] = useState("shap");
  const [topK, setTopK] = useState(10);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const explain = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${apiBase}/api/ml/explain`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, model_name: selectedModel, method, top_k: topK }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.detail || "Explain failed");
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ marginTop: 16 }}>
      <h3>Explain a prediction</h3>
      <textarea value={text} onChange={(e)=>setText(e.target.value)} rows={5} style={{width:"100%"}} placeholder="Paste a news statement..." />
      <div style={{ display:"flex", gap:8, alignItems:"center", marginTop:8 }}>
        <label>Method:</label>
        <select value={method} onChange={(e)=>setMethod(e.target.value)}>
          <option value="shap">SHAP</option>
          <option value="lime">LIME</option>
          <option value="ig">Integrated Gradients</option>
        </select>
        <label>Top‑K:</label>
        <input type="number" min={3} max={30} value={topK} onChange={(e)=>setTopK(parseInt(e.target.value||"10"))} />
        <button onClick={explain} disabled={loading || !text.trim()}>Explain</button>
      </div>
      {error && <p style={{ color:"crimson" }}>{error}</p>}
      {loading && <p>Computing explanation…</p>}
      {result && (
        <div style={{ marginTop: 12 }}>
          <p><b>Method:</b> {result.method}</p>
          {result.predicted_class && <p><b>Predicted:</b> {result.predicted_class}</p>}
          <div style={{ lineHeight: "2.0" }}>
            {result.tokens.map((t, i) => {
              const s = result.scores[i];
              const opacity = Math.min(1, Math.abs(s) / (Math.abs(result.scores[0]) + 1e-6));
              const bg = s >= 0 ? `rgba(0, 128, 0, ${opacity})` : `rgba(220, 20, 60, ${opacity})`;
              return <span key={i} style={{ background:bg, padding:"2px 4px", marginRight:4, borderRadius:3 }}>{t}</span>
            })}
          </div>
        </div>
      )}
    </div>
  );
}
