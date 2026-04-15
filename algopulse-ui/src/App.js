import { useState, useRef } from "react";
import axios from "axios";
import "./App.css";

const BACKEND_URL = "http://localhost:5000";

const USERS = {
  doctor:  { password: "doctor123",  role: "doctor"  },
  patient: { password: "patient123", role: "patient" },
};

function speak(text) {
  if (!window.speechSynthesis) return;
  window.speechSynthesis.cancel();
  const utter = new SpeechSynthesisUtterance(text);
  utter.rate = 0.9; utter.pitch = 1; utter.volume = 1;
  window.speechSynthesis.speak(utter);
}

function buildVoiceMessage(label, confidence) {
  const pct = (confidence * 100).toFixed(0);
  if (label === "Malignant")
    return `Alert. The analysis indicates a malignant tumor with ${pct} percent confidence. This means there may be signs of cancer in the scan. Please consult your doctor immediately for further tests and guidance. Do not panic. Your doctor is here to help you.`;
  if (label === "Benign")
    return `Good news. The analysis indicates a benign tumor with ${pct} percent confidence. This means the growth does not appear to be cancerous. However, please still consult your doctor for confirmation and a follow-up checkup.`;
  if (label === "Normal")
    return `The analysis indicates a normal result with ${pct} percent confidence. No abnormal growth was detected in the scan. Please continue your regular health checkups as advised by your doctor.`;
  return `The result is inconclusive with ${pct} percent confidence. Please visit your doctor for further examination.`;
}

// LOGIN PAGE
function LoginPage({ onLogin }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError]       = useState("");
  const [selected, setSelected] = useState(null);

  const handleLogin = () => {
    const user = USERS[username.toLowerCase()];
    if (!user || user.password !== password) {
      setError("Incorrect username or password. Please try again.");
      return;
    }
    onLogin(user.role);
  };

  return (
    <div className="login-page">
      <div className="login-bg-circles">
        <div className="circle c1"/><div className="circle c2"/><div className="circle c3"/>
      </div>
      <div className="login-box">
        <div className="login-logo">
          <div className="login-logo-icon">
            <svg width="32" height="32" viewBox="0 0 36 36" fill="none">
              <circle cx="18" cy="18" r="17" stroke="#fff" strokeWidth="2"/>
              <path d="M10 18 Q14 10 18 18 Q22 26 26 18" stroke="#F9A8D4" strokeWidth="2.5" fill="none" strokeLinecap="round"/>
              <circle cx="18" cy="18" r="3" fill="#fff"/>
            </svg>
          </div>
          <div>
            <h1 className="login-title">AlgoPulse</h1>
            <p className="login-subtitle">AI Breast Cancer Detection</p>
          </div>
        </div>

        <p className="login-tagline">Who are you logging in as?</p>

        <div className="role-cards">
          <div className={`role-card ${selected === "doctor" ? "active" : ""}`}
            onClick={() => { setSelected("doctor"); setUsername("doctor"); }}>
            <span className="role-emoji">👨‍⚕️</span>
            <span className="role-name">Doctor</span>
            <span className="role-desc">Clinical dashboard with full analysis</span>
          </div>
          <div className={`role-card ${selected === "patient" ? "active" : ""}`}
            onClick={() => { setSelected("patient"); setUsername("patient"); }}>
            <span className="role-emoji">🧑‍💼</span>
            <span className="role-name">Patient</span>
            <span className="role-desc">Simple results with voice explanation</span>
          </div>
        </div>

        <div className="login-field">
          <label className="field-label">Password</label>
          <input className="field-input" type="password" placeholder="Enter your password"
            value={password} onChange={(e) => setPassword(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleLogin()}/>
        </div>

        {error && <div className="login-error">{error}</div>}

        <button className="login-btn" onClick={handleLogin} disabled={!selected}>
          Login as {selected ? (selected === "doctor" ? "Doctor" : "Patient") : "..."}
        </button>

        <p className="login-hint">
          Doctor: <b>doctor</b> / <b>doctor123</b> &nbsp;|&nbsp; Patient: <b>patient</b> / <b>patient123</b>
        </p>
      </div>
    </div>
  );
}

// DOCTOR PAGE
function DoctorPage({ onLogout }) {
  const [image, setImage]       = useState(null);
  const [preview, setPreview]   = useState(null);
  const [result, setResult]     = useState(null);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const fileRef                 = useRef();

  const handleFile = (file) => {
    if (!file) return;
    setImage(file); setPreview(URL.createObjectURL(file));
    setResult(null); setError(null);
  };

  const handleSubmit = async () => {
    if (!image) { setError("Please upload an ultrasound image first."); return; }
    setLoading(true); setError(null); setResult(null);
    const formData = new FormData();
    formData.append("ultrasound", image);
    try {
      const res = await axios.post(`${BACKEND_URL}/predict`, formData,
        { headers: { "Content-Type": "multipart/form-data" } });
      setResult(res.data);
    } catch (err) {
      setError(err.response?.data?.error || "Server error. Make sure backend is running.");
    } finally { setLoading(false); }
  };

  const reset = () => { setImage(null); setPreview(null); setResult(null); setError(null); };
  const pct   = (v) => `${(v * 100).toFixed(1)}%`;

  const labelStyle = (label) => {
    if (label === "Malignant") return { color: "#e53e3e", background: "#fff5f5", border: "2px solid #e53e3e" };
    if (label === "Benign")    return { color: "#1FA4A9", background: "#e6fafa", border: "2px solid #1FA4A9" };
    if (label === "Normal")    return { color: "#38a169", background: "#f0fff4", border: "2px solid #38a169" };
    return {};
  };

  return (
    <div className="app">
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
              <p className="logo-sub">Clinical Dashboard · Doctor View</p>
            </div>
          </div>
          <div style={{display:"flex",gap:"10px",alignItems:"center"}}>
            <button className="logout-btn" onClick={onLogout}>Logout</button>
          </div>
        </div>
      </header>

     <div className="hero-strip">
  <div className="hero-inner">
    <div className="hero-stat">
      <span className="stat-num">⚡</span>
      <span className="stat-label">Instant Results</span>
    </div>
    <div className="hero-divider"/>
    <div className="hero-stat">
      <span className="stat-num">🧠</span>
      <span className="stat-label">AI-Powered Analysis</span>
    </div>
    <div className="hero-divider"/>
    <div className="hero-stat">
      <span className="stat-num">👁️</span>
      <span className="stat-label">Visual Explanations</span>
    </div>
    <div className="hero-divider"/>
    <div className="hero-stat">
      <span className="stat-num">🔒</span>
      <span className="stat-label">Private & Secure</span>
    </div>
  </div>
</div>

      <main className="main">
        <div className="main-grid">
          <section className="card upload-card">
            <div className="card-header">
              <div className="card-icon teal">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                  <polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>
                </svg>
              </div>
              <h2 className="card-title">Upload Ultrasound Image</h2>
            </div>

            <div className={`dropzone ${dragOver?"dragover":""} ${preview?"has-image":""}`}
              onClick={() => fileRef.current.click()}
              onDrop={(e) => { e.preventDefault(); setDragOver(false); handleFile(e.dataTransfer.files[0]); }}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}>
              <input ref={fileRef} type="file" accept="image/*" onChange={(e) => handleFile(e.target.files[0])} hidden/>
              {preview ? (
                <div className="preview-wrapper">
                  <img src={preview} alt="Uploaded" className="preview-img"/>
                  <div className="preview-overlay"><span>Click to change</span></div>
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
                  <p className="drop-sub">JPG, PNG · Ultrasound images only</p>
                </div>
              )}
            </div>

            {image && (
              <div className="file-info">
                <span>{image.name}</span>
                <span className="file-size">({(image.size/1024).toFixed(1)} KB)</span>
              </div>
            )}
            {error && <div className="alert alert-error">{error}</div>}

            <div className="btn-row">
              <button className="btn btn-primary" onClick={handleSubmit} disabled={loading||!image}>
                {loading ? <><span className="spinner"/>&nbsp;Analysing...</> : "Run Diagnosis"}
              </button>
              {(preview||result) && <button className="btn btn-ghost" onClick={reset}>Reset</button>}
            </div>

            <div className="info-box">
              <p className="info-title">ℹ️ How it works</p>
              <p className="info-text">EfficientNet-B0 classifies scans as <b>Normal</b>, <b>Benign</b>, or <b>Malignant</b> and highlights regions using Grad-CAM heatmaps.</p>
            </div>
          </section>

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
                <p className="empty-title">No results yet</p>
                <p className="empty-sub">Upload an image and click Run Diagnosis</p>
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
                <div className="label-pill" style={labelStyle(result.label)}>
                  <span className="label-text">{result.label}</span>
                  <span className="label-conf">{pct(result.confidence)} confidence</span>
                </div>

                {result.flagged && <div className="alert alert-warn">{result.flag_message}</div>}

                <div className="prob-section">
                  <p className="section-label">Class Probabilities</p>
                  {Object.entries(result.probabilities).map(([cls, val]) => (
                    <div key={cls} className="prob-row">
                      <span className="prob-name">{cls}</span>
                      <div className="prob-bar-bg">
                        <div className={`prob-bar-fill ${cls.toLowerCase()}`} style={{width:pct(val)}}/>
                      </div>
                      <span className="prob-pct">{pct(val)}</span>
                    </div>
                  ))}
                </div>

                {result.heatmap && (
                  <div className="heatmap-section">
                    <p className="section-label">Grad-CAM Heatmap</p>
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

                <div className="recommendation">
                  <p className="rec-label">Clinical Recommendation</p>
                  <p className="rec-text">{result.recommendation}</p>
                </div>

                {result.similar_cases && result.similar_cases.length > 0 && (
                  <div className="similar-cases-section">
                    <p className="section-label">📊 Similar Cases in Database</p>
                    <div className="similar-cases-grid">
                      {result.similar_cases.map((cas, idx) => (
                        <div key={idx} className="similar-case-card">
                          <div className="case-label">{cas.label}</div>
                          <div className="case-similarity">Similarity: {(cas.similarity_score * 100).toFixed(1)}%</div>
                          <div className="case-path">{cas.image_path.split('\\').pop()}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <div className="disclaimer">{result.disclaimer}</div>
              </div>
            )}
          </section>
        </div>

        <div className="bottom-strip">
          <div className="strip-item"><span className="strip-icon">🔒</span><span>Data stays on device</span></div>
          <div className="strip-item"><span className="strip-icon">👁️</span><span>Grad-CAM </span></div>
          <div className="strip-item"><span className="strip-icon">⚕️</span><span> Not a medical diagnosis</span></div>
        </div>
      </main>

  
    </div>
  );
}

// PATIENT PAGE
function PatientPage({ onLogout }) {
  const [image, setImage]       = useState(null);
  const [preview, setPreview]   = useState(null);
  const [result, setResult]     = useState(null);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState(null);
  const [speaking, setSpeaking] = useState(false);
  const fileRef                 = useRef();

  const handleFile = (file) => {
    if (!file) return;
    setImage(file); setPreview(URL.createObjectURL(file));
    setResult(null); setError(null);
  };

  const handleSubmit = async () => {
    if (!image) { setError("Please choose your scan image first."); return; }
    setLoading(true); setError(null); setResult(null);
    const formData = new FormData();
    formData.append("ultrasound", image);
    try {
      const res = await axios.post(`${BACKEND_URL}/predict`, formData,
        { headers: { "Content-Type": "multipart/form-data" } });
      setResult(res.data);
      const msg = buildVoiceMessage(res.data.label, res.data.confidence);
      setSpeaking(true);
      speak(msg);
      setTimeout(() => setSpeaking(false), 10000);
    } catch {
      setError("Something went wrong. Please ask a nurse or doctor for help.");
    } finally { setLoading(false); }
  };

  const handleSpeak = () => {
    if (!result) return;
    setSpeaking(true);
    speak(buildVoiceMessage(result.label, result.confidence));
    setTimeout(() => setSpeaking(false), 10000);
  };

  const reset = () => {
    setImage(null); setPreview(null); setResult(null); setError(null);
    window.speechSynthesis.cancel(); setSpeaking(false);
  };

  const pct = (v) => `${(v * 100).toFixed(0)}%`;

  const getResultInfo = (label) => {
    if (label === "Malignant") return {
      emoji: "⚠️", color: "#e53e3e", bg: "#fff5f5", border: "#fc8181",
      heading: "The scan shows signs that need attention",
      simple: "Our AI found something in your scan that your doctor should look at right away. This does NOT mean you should panic — your doctor will explain what this means and guide you on the next steps.",
      action: "Please show this result to your doctor today."
    };
    if (label === "Benign") return {
      emoji: "✅", color: "#1FA4A9", bg: "#e6fafa", border: "#81e6d9",
      heading: "The scan looks okay",
      simple: "Our AI found a small growth in your scan, but it does not appear to be harmful. A benign growth is not cancer. However, your doctor should still check this to be sure.",
      action: "Visit your doctor for a routine follow-up checkup."
    };
    if (label === "Normal") return {
      emoji: "🎉", color: "#38a169", bg: "#f0fff4", border: "#9ae6b4",
      heading: "Your scan looks normal",
      simple: "Our AI did not find anything unusual in your scan. This is good news! However, regular checkups are still important for your health.",
      action: "Keep up with your regular health checkups."
    };
    return {
      emoji: "🔍", color: "#d69e2e", bg: "#fffff0", border: "#faf089",
      heading: "We need more information",
      simple: "Our AI was not fully sure about the result. Please see a doctor who can do a proper examination.",
      action: "Please visit a doctor for further tests."
    };
  };

  return (
    <div className="patient-app">
      <header className="patient-header">
        <div className="patient-header-inner">
          <div className="logo-group">
            <div className="login-logo-icon" style={{width:44,height:44}}>
              <svg width="28" height="28" viewBox="0 0 36 36" fill="none">
                <circle cx="18" cy="18" r="17" stroke="#fff" strokeWidth="2"/>
                <path d="M10 18 Q14 10 18 18 Q22 26 26 18" stroke="#F9A8D4" strokeWidth="2.5" fill="none" strokeLinecap="round"/>
                <circle cx="18" cy="18" r="3" fill="#fff"/>
              </svg>
            </div>
            <div>
              <h1 className="logo-title" style={{fontSize:"1.2rem"}}>AlgoPulse</h1>
              <p className="logo-sub">Your Health Assistant</p>
            </div>
          </div>
          <button className="logout-btn" onClick={onLogout}>Logout</button>
        </div>
      </header>

      <main className="patient-main">
        <div className="patient-welcome">
          <h2 className="patient-welcome-title">Hello! 👋</h2>
          <p className="patient-welcome-sub">Upload your breast scan below and we will explain your results in simple words. You can also listen to your results using the voice button.</p>
        </div>

        <div className="patient-card">
          <h3 className="patient-section-title">📁 Step 1 — Upload Your Scan</h3>
          <p className="patient-section-desc">Ask your nurse or doctor to help you upload your ultrasound scan image.</p>

          <div className={`patient-dropzone ${preview ? "has-image" : ""}`}
            onClick={() => fileRef.current.click()}>
            <input ref={fileRef} type="file" accept="image/*"
              onChange={(e) => handleFile(e.target.files[0])} hidden/>
            {preview ? (
              <div className="preview-wrapper">
                <img src={preview} alt="Your scan" className="preview-img" style={{maxHeight:220}}/>
                <div className="preview-overlay"><span>Tap to change</span></div>
              </div>
            ) : (
              <div className="drop-content">
                <div style={{fontSize:"3rem",marginBottom:"0.5rem"}}>📷</div>
                <p style={{fontWeight:600,color:"#2d3748",fontSize:"1rem"}}>Tap here to choose your scan</p>
                <p style={{fontSize:"0.82rem",color:"#718096",marginTop:4}}>JPG or PNG image file</p>
              </div>
            )}
          </div>

          {image && <p className="patient-file-ok">✅ Image ready: {image.name}</p>}
          {error && <div className="patient-error">{error}</div>}

          <button className="patient-btn" onClick={handleSubmit} disabled={loading||!image}>
            {loading ? "🔍 Checking your scan..." : "🔍 Check My Scan"}
          </button>
          {(preview||result) && (
            <button className="patient-btn-ghost" onClick={reset}>Start Again</button>
          )}
        </div>

        {loading && (
          <div className="patient-card" style={{textAlign:"center",padding:"2.5rem"}}>
            <div className="pulse-ring" style={{margin:"0 auto 1rem"}}/>
            <p style={{fontWeight:600,fontSize:"1.1rem",color:"#1FA4A9"}}>Checking your scan...</p>
            <p style={{color:"#718096",fontSize:"0.88rem",marginTop:8}}>This will take a few seconds. Please wait.</p>
          </div>
        )}

        {result && (() => {
          const info = getResultInfo(result.label);
          return (
            <div className="patient-card patient-result-card"
              style={{borderColor:info.border, background:info.bg}}>
              <div className="patient-result-top">
                <span className="patient-result-emoji">{info.emoji}</span>
                <div>
                  <h3 className="patient-result-heading" style={{color:info.color}}>{info.heading}</h3>
                  <p className="patient-confidence-badge">AI is <b>{pct(result.confidence)}</b> sure about this result</p>
                </div>
              </div>

              <p className="patient-result-simple">{info.simple}</p>

              <div className="patient-action-box">
                <span style={{fontSize:"1.2rem"}}>📋</span>
                <p><b>What to do next:</b> {info.action}</p>
              </div>

              <div className="voice-section">
                <button className={`voice-btn ${speaking?"speaking":""}`} onClick={handleSpeak}>
                  <span className="voice-icon">{speaking ? "🔊" : "🔈"}</span>
                  <span>{speaking ? "Speaking... tap to replay" : "Tap to hear your result"}</span>
                </button>
                <p className="voice-hint">
                  🎧 This voice feature helps patients who have difficulty reading, elderly patients, and people in areas with limited healthcare access.
                </p>
              </div>

              {result.heatmap && (
                <div style={{marginTop:"1.5rem"}}>
                  <p className="patient-section-title" style={{fontSize:"1rem"}}>🖼️ What the AI looked at</p>
                  <p style={{fontSize:"0.82rem",color:"#718096",marginBottom:"0.75rem"}}>The coloured areas show the parts of your scan that the AI focused on.</p>
                  <div className="heatmap-grid">
                    <div className="heatmap-item">
                      <img src={preview} alt="Your scan" className="heatmap-img"/>
                      <p className="heatmap-caption">Your scan</p>
                    </div>
                    <div className="heatmap-item">
                      <img src={result.heatmap} alt="Areas checked" className="heatmap-img"/>
                      <p className="heatmap-caption">Areas the AI checked</p>
                    </div>
                  </div>
                </div>
              )}

              <div className="patient-disclaimer">
                ⚕️ <b>Important:</b> This AI tool is here to help, not to replace your doctor. Always talk to a qualified doctor about your results.
              </div>
            </div>
          );
        })()}

        <div className="patient-card accessibility-card">
          <h3 className="patient-section-title">♿ Why we have a voice feature</h3>
          <div className="access-items">
            <div className="access-item">
              <span className="access-icon">👁️</span>
              <div>
                <p className="access-title">Visually impaired patients</p>
                <p className="access-desc">Hear your results clearly without needing to read the screen</p>
              </div>
            </div>
            <div className="access-item">
              <span className="access-icon">👴</span>
              <div>
                <p className="access-title">Elderly patients</p>
                <p className="access-desc">Easy to understand spoken explanation of medical results</p>
              </div>
            </div>
            <div className="access-item">
              <span className="access-icon">🌾</span>
              <div>
                <p className="access-title">Rural healthcare setups</p>
                <p className="access-desc">AI voice bridges the gap where doctors may not be immediately available</p>
              </div>
            </div>
          </div>
        </div>
      </main>

      <footer className="footer" style={{marginTop:"2rem"}}>
        <p>AlgoPulse · Patient Portal</p>
      </footer>
    </div>
  );
}

// ROOT
export default function App() {
  const [role, setRole] = useState(null);
  if (!role)              return <LoginPage onLogin={setRole}/>;
  if (role === "doctor")  return <DoctorPage onLogout={() => setRole(null)}/>;
  return <PatientPage onLogout={() => setRole(null)}/>;
}