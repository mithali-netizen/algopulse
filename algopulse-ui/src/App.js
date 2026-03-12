import { useState, useRef } from "react";
import axios from "axios";
import "./App.css";

const BACKEND_URL = "http://localhost:5000";

export default function App() {
  const [image, setImage]           = useState(null);
  const [preview, setPreview]       = useState(null);
  const [result, setResult]         = useState(null);
  const [loading, setLoading]       = useState(false);
  const [error, setError]           = useState(null);
  const [dragOver, setDragOver]     = useState(false);
  const fileRef                     = useRef();

  // ── Handle file selection ──────────────────────────────────────────────
  const handleFile = (file) => {
    if (!file) return;
    setImage(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  };

  const onFileChange  = (e) => handleFile(e.target.files[0]);
  const onDrop        = (e) => { e.preventDefault(); setDragOver(false); handleFile(e.dataTransfer.files[0]); };
  const onDragOver    = (e) => { e.preventDefault(); setDragOver(true); };
  const onDragLeave   = ()  => setDragOver(false);

  // ── Submit to backend ──────────────────────────────────────────────────
  const handleSubmit = async () => {
    if (!image) { setError("Please upload an ultrasound image first."); return; }
    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("ultrasound", image);

    try {
      const res = await axios.post(`${BACKEND_URL}/predict`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult(res.data);
    } catch (err) {
      setError(err.response?.data?.error || "Server error. Make sure the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  const reset = () => { setImage(null); setPreview(null); setResult(null); setError(null); };

  // ── Label colours ──────────────────────────────────────────────────────
  const labelStyle = (label) => {
    if (label === "Malignant")    return { color: "#e53e3e", background: "#fff5f5", border: "2px solid #e53e3e" };
    if (label === "Benign")       return { color: "#1FA4A9", background: "#e6fafa", border: "2px solid #1FA4A9" };
    if (label === "Normal")       return { color: "#38a169", background: "#f0fff4", border: "2px solid #38a169" };
    if (label === "Inconclusive") return { color: "#d69e2e", background: "#fffff0", border: "2px solid #d69e2e" };
    return {};
  };

  const pct = (v) => `${(v * 100).toFixed(1)}%`;

  // ── Render ─────────────────────────────────────────────────────────────
  return (
    <div className="app">

      {/* ── HEADER ── */}
      <header className="header">
        <div className="header-inner">
          <div className="logo-group">
            <div className="logo-icon">
              <svg width="36" height="36" viewBox="0 0 36 36" fill="none">
                <circle cx="18" cy="18" r="17" stroke="#fff" strokeWidth="2"/>
                <path d="M10 18 Q14 10 18 18 Q22 26 26 18" stroke="#F9A8D4" strokeWidth="2.5" fill="none" strokeLinecap="round"/>
                <circle cx="18" cy="18" r="3" fill="#fff"/>
              </svg>
            </div>
            <div>
              <h1 className="logo-title">AlgoPulse</h1>
              <p className="logo-sub">AI Breast Cancer Detection</p>
            </div>
          </div>
          <div className="header-badge">
            <span className="badge-dot"/> ETHiCARE AI 2026 · Track 1-3
          </div>
        </div>
      </header>

      {/* ── HERO STRIP ── */}
      <div className="hero-strip">
        <div className="hero-inner">
          <div className="hero-stat"><span className="stat-num">3</span><span className="stat-label">Classes</span></div>
          <div className="hero-divider"/>
          <div className="hero-stat"><span className="stat-num">EfficientNet-B0</span><span className="stat-label">Model Architecture</span></div>
          <div className="hero-divider"/>
          <div className="hero-stat"><span className="stat-num">Grad-CAM</span><span className="stat-label">Explainability</span></div>
          <div className="hero-divider"/>
          <div className="hero-stat"><span className="stat-num">Transfer</span><span className="stat-label">Learning</span></div>
        </div>
      </div>

      {/* ── MAIN ── */}
      <main className="main">
        <div className="main-grid">

          {/* ── LEFT PANEL — Upload ── */}
          <section className="card upload-card">
            <div className="card-header">
              <div className="card-icon teal">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                  <polyline points="17 8 12 3 7 8"/>
                  <line x1="12" y1="3" x2="12" y2="15"/>
                </svg>
              </div>
              <h2 className="card-title">Upload Ultrasound Image</h2>
            </div>

            {/* Drop zone */}
            <div
              className={`dropzone ${dragOver ? "dragover" : ""} ${preview ? "has-image" : ""}`}
              onClick={() => fileRef.current.click()}
              onDrop={onDrop}
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
            >
              <input ref={fileRef} type="file" accept="image/*" onChange={onFileChange} hidden/>

              {preview ? (
                <div className="preview-wrapper">
                  <img src={preview} alt="Uploaded ultrasound" className="preview-img"/>
                  <div className="preview-overlay">
                    <span>Click to change</span>
                  </div>
                </div>
              ) : (
                <div className="drop-content">
                  <div className="drop-icon">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#1FA4A9" strokeWidth="1.5">
                      <rect x="3" y="3" width="18" height="18" rx="2"/>
                      <circle cx="8.5" cy="8.5" r="1.5"/>
                      <polyline points="21 15 16 10 5 21"/>
                    </svg>
                  </div>
                  <p className="drop-title">Drag & drop or click to upload</p>
                  <p className="drop-sub">JPG, PNG accepted · Ultrasound images only</p>
                </div>
              )}
            </div>

            {image && (
              <div className="file-info">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#1FA4A9" strokeWidth="2">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                  <polyline points="14 2 14 8 20 8"/>
                </svg>
                <span>{image.name}</span>
                <span className="file-size">({(image.size / 1024).toFixed(1)} KB)</span>
              </div>
            )}

            {error && (
              <div className="alert alert-error">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="10"/>
                  <line x1="12" y1="8" x2="12" y2="12"/>
                  <line x1="12" y1="16" x2="12.01" y2="16"/>
                </svg>
                {error}
              </div>
            )}

            <div className="btn-row">
              <button className="btn btn-primary" onClick={handleSubmit} disabled={loading || !image}>
                {loading ? (
                  <><span className="spinner"/>&nbsp;Analysing...</>
                ) : (
                  <><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>&nbsp;Run Diagnosis</>
                )}
              </button>
              {(preview || result) && (
                <button className="btn btn-ghost" onClick={reset}>Reset</button>
              )}
            </div>

            {/* Info box */}
            <div className="info-box">
              <p className="info-title">ℹ️ How it works</p>
              <p className="info-text">Upload a breast ultrasound scan. Our EfficientNet-B0 model classifies it as <b>Normal</b>, <b>Benign</b>, or <b>Malignant</b> and highlights suspicious regions using Grad-CAM heatmaps.</p>
            </div>
          </section>

          {/* ── RIGHT PANEL — Results ── */}
          <section className="card result-card">
            <div className="card-header">
              <div className="card-icon pink">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
                </svg>
              </div>
              <h2 className="card-title">Diagnostic Results</h2>
            </div>

            {!result && !loading && (
              <div className="empty-state">
                <div className="empty-icon">
                  <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="#CBD5E0" strokeWidth="1">
                    <path d="M9 3H5a2 2 0 0 0-2 2v4m6-6h10a2 2 0 0 1 2 2v4M9 3v18m0 0h10a2 2 0 0 0 2-2V9M9 21H5a2 2 0 0 1-2-2V9m0 0h18"/>
                  </svg>
                </div>
                <p className="empty-title">No results yet</p>
                <p className="empty-sub">Upload an ultrasound image and click <b>Run Diagnosis</b></p>
              </div>
            )}

            {loading && (
              <div className="loading-state">
                <div className="pulse-ring"/>
                <p className="loading-title">Analysing image...</p>
                <p className="loading-sub">Running EfficientNet-B0 · Generating Grad-CAM</p>
              </div>
            )}

            {result && (
              <div className="results-body">

                {/* Label pill */}
                <div className="label-pill" style={labelStyle(result.label)}>
                  <span className="label-text">{result.label}</span>
                  <span className="label-conf">{pct(result.confidence)} confidence</span>
                </div>

                {/* Flag warning */}
                {result.flagged && (
                  <div className="alert alert-warn">{result.flag_message}</div>
                )}

                {/* Probability bars */}
                <div className="prob-section">
                  <p className="section-label">Class Probabilities</p>
                  {Object.entries(result.probabilities).map(([cls, val]) => (
                    <div key={cls} className="prob-row">
                      <span className="prob-name">{cls}</span>
                      <div className="prob-bar-bg">
                        <div
                          className={`prob-bar-fill ${cls.toLowerCase()}`}
                          style={{ width: pct(val) }}
                        />
                      </div>
                      <span className="prob-pct">{pct(val)}</span>
                    </div>
                  ))}
                </div>

                {/* Heatmap */}
                {result.heatmap && (
                  <div className="heatmap-section">
                    <p className="section-label">Grad-CAM Heatmap
                      <span className="section-hint"> — highlighted regions influenced the prediction</span>
                    </p>
                    <div className="heatmap-grid">
                      <div className="heatmap-item">
                        <img src={preview} alt="Original" className="heatmap-img"/>
                        <p className="heatmap-caption">Original</p>
                      </div>
                      <div className="heatmap-item">
                        <img src={result.heatmap} alt="Grad-CAM" className="heatmap-img"/>
                        <p className="heatmap-caption">Grad-CAM Overlay</p>
                      </div>
                    </div>
                  </div>
                )}

                {/* Recommendation */}
                <div className="recommendation">
                  <p className="rec-label">Clinical Recommendation</p>
                  <p className="rec-text">{result.recommendation}</p>
                </div>

                {/* Disclaimer */}
                <div className="disclaimer">
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="10"/>
                    <line x1="12" y1="8" x2="12" y2="12"/>
                    <line x1="12" y1="16" x2="12.01" y2="16"/>
                  </svg>
                  {result.disclaimer}
                </div>

              </div>
            )}
          </section>

        </div>

        {/* ── BOTTOM INFO STRIP ── */}
        <div className="bottom-strip">
          <div className="strip-item">
            <span className="strip-icon">🔒</span>
            <span>Patient data never leaves your device</span>
          </div>
          <div className="strip-item">
            <span className="strip-icon">🧠</span>
            <span>EfficientNet-B0 with transfer learning</span>
          </div>
          <div className="strip-item">
            <span className="strip-icon">👁️</span>
            <span>Grad-CAM explainability built-in</span>
          </div>
          <div className="strip-item">
            <span className="strip-icon">⚕️</span>
            <span>Decision-support tool — clinician always decides</span>
          </div>
        </div>

      </main>

      {/* ── FOOTER ── */}
      <footer className="footer">
        <p>AlgoPulse · ETHiCARE AI 2026 · Track 1-3 · Department of Medical Electronics Engineering, DSCE</p>
        <p className="footer-sub">Built with PyTorch · Flask · React · EfficientNet-B0 · Grad-CAM</p>
      </footer>

    </div>
  );
}
