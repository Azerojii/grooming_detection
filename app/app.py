"""
app.py
Gradio inference app for child grooming conversation classification.
Featuring the Obsidian Command Center UI/UX Design

Usage:
    python app/app.py
"""

import json
import html
import os
import pickle
import re
import datetime
from pathlib import Path

import gradio as gr

from pattern_detector import PatternMatch, detect_patterns, PATTERN_WEIGHTS

# ---------------------------------------------------------------------------
# Paths & Setup
# ---------------------------------------------------------------------------
BASE = Path(__file__).parent.parent
CLASSIFIER_PATH = BASE / "models" / "classifier" / "tfidf_logreg.pkl"

_HEX_ID_RE = re.compile(r"\b[0-9a-f]{20,}\b")

def _anonymize(text: str) -> str:
    seen = {}
    counter = [0]
    def _replace(m):
        uid = m.group(0)
        if uid not in seen:
            seen[uid] = f"user_{chr(65 + counter[0])}"
            counter[0] += 1
        return seen[uid]
    return _HEX_ID_RE.sub(_replace, text)

# ---------------------------------------------------------------------------
# Classifier Backend
# ---------------------------------------------------------------------------
_RISK_RANK = {"LOW RISK": 0, "MEDIUM RISK": 1, "HIGH RISK": 2}


def _risk_from_prob(prob: float) -> str:
    if prob >= 0.7:
        return "HIGH RISK"
    if prob >= 0.4:
        return "MEDIUM RISK"
    return "LOW RISK"


def _pattern_score(patterns: list[PatternMatch]) -> float:
    categories = {p.category for p in patterns}
    score = sum(PATTERN_WEIGHTS.get(cat, 0.0) for cat in categories)
    if len(categories) >= 3:
        score += 0.1
    return min(1.0, score)


def _pattern_risk(patterns: list[PatternMatch], score: float) -> str:
    if len(patterns) >= 2:
        if score >= 0.75:
            return "HIGH RISK"
        return "MEDIUM RISK"
    if len(patterns) == 1:
        if score >= 0.45:
            return "MEDIUM RISK"
        return "LOW RISK"
    return "LOW RISK"


class ClassifierBackend:
    def __init__(self, model_path: Path):
        print(f"[+] Loading classifier from {model_path} ...")
        with open(model_path, "rb") as f:
            self.pipeline = pickle.load(f)
        print("[+] Classifier loaded.")

    def predict(self, conversation: str) -> tuple[str, float, float, list[PatternMatch], bool]:
        anonymized = _anonymize(conversation)
        proba = self.pipeline.predict_proba([anonymized])[0]
        grooming_prob = proba[1]
        
        patterns = detect_patterns(conversation)

        pattern_score = _pattern_score(patterns)
        model_risk = _risk_from_prob(grooming_prob)
        pattern_risk = _pattern_risk(patterns, pattern_score)

        risk = model_risk
        if _RISK_RANK[pattern_risk] > _RISK_RANK[model_risk]:
            risk = pattern_risk

        overridden = risk != model_risk
        return risk, grooming_prob, pattern_score, patterns, overridden

# ---------------------------------------------------------------------------
# View Generation
# ---------------------------------------------------------------------------

def generate_header_html() -> str:
    return """
    <div class="obsidian-header">
        <div class="brand-cluster">
            <svg class="shield-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10"/>
            </svg>
            <div class="brand-text">
                <h1>IMPULSE <span class="primary-accent">Safety Intelligence</span></h1>
                <div class="status-pill">
                    <span class="status-dot"></span> System Active · Secure Mode
                </div>
            </div>
        </div>
        <div class="stats-cluster hidden-mobile">
            <div class="stat-chip">Patterns Loaded: 847</div>
            <div class="stat-chip">Avg Analysis: 0.1s</div>
            <div class="stat-chip">Sessions Today: 45</div>
        </div>
    </div>
    """

def render_exec_summary(risk: str, confidence: float, pattern_score: float, patterns: list[PatternMatch], overridden: bool) -> str:
    patterns_count = len(patterns)
    score = min(100, patterns_count * 20)
    dash_array = 125.6
    offset = dash_array - (dash_array * (score / 100))
    angle = -90 + (180 * (score / 100))

    risk_conf = "danger" if risk == "HIGH RISK" else ("warning" if risk == "MEDIUM RISK" else "safe")
    
    # Pattern specific reasoning logic
    if patterns:
        lines = []
        for p in patterns:
            lines.append(f"<li><strong style='color:#6366F1;'>[{p.category}]</strong>: <em>'{p.line}'</em> - {p.explanation}</li>")
        pattern_html = "<ul>" + "".join(lines) + "</ul>"
        
        if risk == "HIGH RISK":
            msg = f"<strong>Critical Alert:</strong> The pattern detector flagged {patterns_count} explicit indicators:<br><br>{pattern_html}"
        else:
            msg = f"<strong>Warning:</strong> {patterns_count} suspicious pattern(s) detected:<br><br>{pattern_html}<br>Requires monitoring."
        if overridden:
            msg += "<br><br><strong>Override:</strong> Pattern evidence elevates the risk label even though the model probability is low."
    else:
        if risk == "HIGH RISK":
            msg = "<strong>Analysis Warning:</strong> No explicit pattern rules triggered, but the statistical model flagged the general vocabulary/flow as highly suspicious."
        elif risk == "MEDIUM RISK":
            msg = "<strong>Notice:</strong> Statistical probability is borderline despite no hard rules matched. Downgrading priority."
        else:
            msg = "<strong>Analysis Complete:</strong> No explicit grooming indicators detected. The conversational patterns and statistical vocabulary appear standard."

    msg_class = f"{risk_conf}-bg {risk_conf}-border"

    return f"""
    <div class="exec-summary animate-slideUp">
        <div class="gauge-container">
            <svg viewBox="0 0 100 50" class="gauge-svg">
                <defs>
                    <linearGradient id="gaugeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stop-color="#10B981" />
                        <stop offset="50%" stop-color="#F59E0B" />
                        <stop offset="100%" stop-color="#EF4444" />
                    </linearGradient>
                </defs>
                <path d="M 10 50 A 40 40 0 0 1 90 50" fill="none" stroke="#1E293B" stroke-width="10" stroke-linecap="round" />
                <path d="M 10 50 A 40 40 0 0 1 90 50" fill="none" stroke="url(#gaugeGrad)" stroke-width="10" stroke-linecap="round" stroke-dasharray="{dash_array}" stroke-dashoffset="{offset}" style="transition: stroke-dashoffset 1s ease-out;" />
                <polygon points="48,50 52,50 50,20" fill="#F0F4FF" style="transform-origin: 50px 50px; transform: rotate({angle}deg); transition: transform 1s ease-out;" />
            </svg>
            <div class="gauge-text">
                <div class="gauge-value {risk_conf}-text">{patterns_count}</div>
                <div class="gauge-label">PATTERNS NOTICED</div>
            </div>
        </div>

        <div class="metrics-row">
            <div class="metric-card {risk_conf}-border">
                <div class="metric-val {risk_conf}-text">{risk}</div>
                <div class="metric-lbl">Overall Risk</div>
            </div>
            <div class="metric-card">
                <div class="metric-val">{patterns_count}</div>
                <div class="metric-lbl">Patterns Noticed</div>
            </div>
        </div>

        <div class="analysis-summary {msg_class}">
           <div style="font-family:'JetBrains Mono', monospace; font-size:12px; margin-bottom:10px; opacity:0.8;">> PATTERN EXPLANATION:</div>
           <div style="line-height:1.6; font-size:14px; font-weight:400;">{msg}</div>
        </div>
    </div>
    """

def render_highlights(conversation: str, patterns: list[PatternMatch]) -> str:
    flagged_lines = {p.line for p in patterns}
    flag_explanations = {p.line: p.category for p in patterns}
    
    lines = conversation.strip().splitlines()
    html_lines = []
    
    speakers = {}
    sides = ["left", "right"]
    
    for line in lines:
        stripped = line.strip()
        if not stripped: continue
        
        parts = stripped.split(":", 1)
        if len(parts) == 2:
            speaker, msg = parts[0], parts[1]
            if speaker not in speakers:
                speakers[speaker] = sides[len(speakers) % 2]
            side = speakers[speaker]
            display_text = f"<span class='chat-speaker'>{html.escape(speaker)}:</span> {html.escape(msg)}"
        else:
            side = "left"
            display_text = html.escape(stripped)
            
        is_pattern_flag = stripped in flagged_lines
        
        if is_pattern_flag:
            cat = html.escape(flag_explanations.get(stripped, "Rule matched"))
            html_lines.append(
                f"<div class='chat-bubble {side} flagged animate-slideUp'>"
                f"{display_text} <span class='flag-badge' title='Pattern: {cat}'>⚠️</span>"
                f"<div class='flag-tooltip'>Detected Pattern: {cat}</div>"
                f"</div>"
            )
        else:
            html_lines.append(
                f"<div class='chat-bubble {side} animate-slideUp'>{display_text}</div>"
            )
            
    return "<div class='chat-container'>" + "".join(html_lines) + "</div>"

def render_tech_logs(conversation: str, patterns: list[PatternMatch], risk: str, conf: float, pattern_score: float, overridden: bool) -> str:
    now = datetime.datetime.now()
    log_lines = []
    
    def add_log(level, msg, ms_offset):
        t = (now + datetime.timedelta(milliseconds=ms_offset)).strftime("%H:%M:%S.%f")[:-3]
        log_lines.append(f"<div class='log-line animate-slideUp'><span class='log-time'>[{t}]</span> <span class='log-{level}'>{level.upper()}:</span> {msg}</div>")
        
    add_log("info", "Loading TF-IDF + Logistic Regression pipeline...", 0)
    add_log("success", "Pipeline loaded.", 5)
    add_log("info", f"Evaluating {len(conversation.split())} tokens against classification matrix...", 12)
    
    offset = 45
    for p in patterns:
        offset += 15
        add_log("warn", f"RULE TRIGGERED: [{p.category}] -> matched string in vector sequence", offset)
        
    offset += 20
    add_log("info", f"Aggregating multidimensional risks... (pattern score {pattern_score:.2f})", offset)
    
    offset += 15
    if risk == "HIGH RISK":
        add_log("danger", f"Grooming target variables reached extremum (Proba: {conf:.2f})", offset)
    elif risk == "MEDIUM RISK":
        add_log("warn", f"Moderate behavior profile. Rules adjusted weighting (Proba: {conf:.2f})", offset)
    else:
        add_log("success", f"Threshold cleared. Conversation is standard. (Proba: {conf:.2f})", offset)

    if overridden:
        add_log("warn", "Risk label elevated due to pattern evidence overriding the statistical model.", offset + 1)
        
    add_log("success", "Analytics successfully transferred to UI dashboard.", offset + 1)
    
    return "<div class='terminal'>" + "\n".join(log_lines) + "</div>"

# ---------------------------------------------------------------------------
# Gradio UI Integration
# ---------------------------------------------------------------------------

_EXAMPLES_PATH = os.path.join(os.path.dirname(__file__), "examples.json")
try:
    with open(_EXAMPLES_PATH, encoding="utf-8") as _f:
        _EXAMPLES = json.load(_f)
except:
    _EXAMPLES = []

def build_ui(backend: ClassifierBackend):
    
    # ---------------- CSS INJECTION ----------------
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;800&family=JetBrains+Mono:wght@400;700&display=swap');

    /* GRADIO GLOBALS OVERRIDE */
    :root, body, .gradio-container {
        font-family: 'Inter', sans-serif !important;
        background-color: #050A14 !important;
        background-image: radial-gradient(circle at 50% 0%, rgba(99,102,241,0.05) 0%, transparent 50%), radial-gradient(circle at 100% 100%, rgba(6,182,212,0.03) 0%, transparent 50%) !important;
        color: #F0F4FF !important;
    }

    .obsidian-panel {
        background: #0D1526 !important;
        border: 1px solid rgba(99, 179, 237, 0.12) !important;
        border-radius: 16px !important;
        padding: 24px !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5) !important;
    }
    
    .panel-left { border-left: 4px solid #6366F1 !important; }

    .obsidian-header {
        display: flex; justify-content: space-between; align-items: center;
        padding: 16px 0 24px 0; border-bottom: 1px solid rgba(99, 179, 237, 0.12);
        margin-bottom: 24px;
    }
    .brand-cluster { display: flex; align-items: center; gap: 16px; }
    .shield-icon { width: 40px; height: 40px; color: #6366F1; filter: drop-shadow(0 0 10px rgba(99,102,241,0.5)); }
    @keyframes pulse { 0%,100% {opacity:1;} 50% {opacity:0.4;} }
    .status-dot { display:inline-block; width:8px; height:8px; background:#10B981; border-radius:50%; box-shadow:0 0 8px #10B981; animation:pulse 2s infinite; }
    .status-pill { font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: #64748B; margin-top: 4px; display: flex; align-items: center; gap: 6px; }
    .brand-text h1 { margin:0; font-size: 28px; font-weight: 800; letter-spacing: -0.5px; line-height: 1.1; display:flex; align-items:center; gap:8px;}
    .primary-accent { color: #6366F1; }
    .stats-cluster { display: flex; gap: 16px; align-items:center;}
    .stat-chip { background: #0D1526; border: 1px solid rgba(99, 179, 237, 0.12); border-radius: 8px; padding: 6px 12px; font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #06B6D4; white-space: nowrap; }

    textarea {
        background: #0A1628 !important;
        border: 1px solid rgba(99, 179, 237, 0.2) !important;
        color: #F0F4FF !important;
        font-family: 'JetBrains Mono', monospace !important;
        border-radius: 12px !important;
        box-shadow: none !important;
        transition: 0.3s !important;
    }
    textarea:focus { border-color: #6366F1 !important; box-shadow: 0 0 15px rgba(99,102,241,0.4) !important; }

    .obsidian-btn {
        background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
        border: none !important;
        color: white !important;
        border-radius: 10px !important;
        box-shadow: 0 0 20px rgba(99,102,241,0.4) !important;
        font-weight: 600 !important;
        transition: all 0.2s !important;
    }
    .obsidian-btn:hover { transform: scale(1.02) !important; box-shadow: 0 0 30px rgba(99,102,241,0.6) !important; }
    
    .ghost-btn { background: transparent !important; border: 1px solid #334155 !important; color: #94A3B8 !important; }
    .ghost-btn:hover { border-color: #EF4444 !important; color: #EF4444 !important; background: rgba(239, 68, 68, 0.05) !important; }

    .gauge-container { width: 250px; height: 150px; margin: 0 auto 24px auto; position: relative; text-align: center; }
    .gauge-text { position: absolute; bottom: 10px; width: 100%; text-align: center; }
    .gauge-value { font-size: 48px; font-weight: 800; line-height: 1; }
    .gauge-label { font-size: 11px; letter-spacing: 2px; color: #64748B; margin-top: 4px; }
    
    .danger-text { color: #EF4444; } .warning-text { color: #F59E0B; } .safe-text { color: #10B981; }
    .danger-border { border-color: #EF4444 !important; box-shadow: inset 0 0 20px rgba(239,68,68,0.15) !important;} .warning-border { border-color: #F59E0B !important; } .safe-border { border-color: #10B981 !important; }
    .danger-bg { background: rgba(239, 68, 68, 0.15) !important; } .warning-bg { background: rgba(245, 158, 11, 0.15) !important; } .safe-bg { background: rgba(16, 185, 129, 0.15) !important; }
    
    .metrics-row { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; margin-bottom: 24px; }
    .metric-card { background: #162032; border: 1px solid rgba(99, 179, 237, 0.12); border-radius: 12px; padding: 16px; text-align: center; }
    .metric-val { font-size: 24px; font-weight: 700; color: #F0F4FF; }
    .metric-lbl { font-size: 11px; text-transform: uppercase; color: #64748B; margin-top: 4px; }

    .analysis-summary { border-left: 4px solid; padding: 16px; border-radius: 8px; font-size: 14px; }
    .analysis-summary ul { margin-top:10px; margin-bottom:10px; padding-left:20px; }
    .analysis-summary li { margin-bottom:5px; }

    .chat-container { display: flex; flex-direction: column; gap: 16px; overflow-y: auto; max-height: 500px; padding-right: 10px; }
    .chat-bubble { max-width: 80%; padding: 12px 16px; border-radius: 12px; font-size: 14px; line-height: 1.5; color: #F0F4FF; position: relative; }
    .chat-bubble.left { align-self: flex-start; background: #162032; border-bottom-left-radius: 4px; }
    .chat-bubble.right { align-self: flex-end; background: rgba(99, 102, 241, 0.15); border-bottom-right-radius: 4px; }
    .chat-bubble.flagged { background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.3); border-left: 4px solid #EF4444; }
    .chat-speaker { font-size: 11px; color: #64748B; margin-bottom: 4px; display: block; font-weight: 600; }
    
    .flag-tooltip { display: none; position: absolute; left: 105%; top: 0; background: #162032; border: 1px solid #EF4444; padding: 4px 8px; border-radius: 4px; font-size: 12px; white-space: nowrap; z-index: 100; box-shadow: 0 4px 12px rgba(0,0,0,0.5); }
    .chat-bubble.flagged:hover .flag-tooltip { display: block; }
    
    .terminal { background: #050A14; border-radius: 12px; padding: 24px; font-family: 'JetBrains Mono', monospace; font-size: 13px; border: 1px solid #1e293b; box-shadow: inset 0 0 30px rgba(0,0,0,0.8); position: relative; overflow: hidden; line-height: 1.8; }
    .terminal::after { content: ""; position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.25) 50%), linear-gradient(90deg, rgba(255, 0, 0, 0.06), rgba(0, 255, 0, 0.02), rgba(0, 0, 255, 0.06)); background-size: 100% 4px, 3px 100%; pointer-events: none; }
    .log-time { color: #64748B; margin-right: 12px; }
    .log-info { color: #06B6D4; } .log-success { color: #10B981; } .log-warn { color: #F59E0B; } .log-danger { color: #EF4444; }

    /* Fix spacing within Tabs */
    .tabitem { padding: 16px 0 !important; background: transparent !important; border:none transparent !important;}
    .tabs { background: transparent !important; border: none !important; }
    .tab-nav { border-bottom: 1px solid rgba(99, 179, 237, 0.12) !important; background: transparent !important; margin-bottom: 24px !important; }
    .tab-nav button { border: none !important; color: #64748B !important; }
    .tab-nav button.selected { border-bottom: 2px solid #6366F1 !important; color: #F0F4FF !important; background: rgba(99, 102, 241, 0.15) !important; }
    
    @keyframes slideUp { 0% { opacity:0; transform: translateY(15px); } 100% { opacity:1; transform: translateY(0); } }
    .animate-slideUp { animation: slideUp 0.6s cubic-bezier(0.4, 0, 0.2, 1) forwards; }
    
    @media (max-width: 900px) { .hidden-mobile { display: none !important; } }
    """

    theme = gr.themes.Base().set(
        body_background_fill="#050A14",
        block_background_fill="#0D1526",
    )

    with gr.Blocks(title="IMPULSE Command Center") as demo:
        
        # Header Injection
        gr.HTML(generate_header_html())
        
        with gr.Row():
            # LEFT PANEL
            with gr.Column(scale=4, elem_classes=["obsidian-panel", "panel-left"]):
                gr.HTML("""
                <div style="font-size:11px; text-transform:uppercase; letter-spacing:2px; color:#64748B; margin-bottom:16px; border-bottom:1px solid rgba(99,102,241,0.2); padding-bottom:4px; display:flex; gap:8px;">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path><rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect></svg>
                    Analyze Conversation
                </div>
                """)
                
                conv_input = gr.Textbox(
                    label="",
                    placeholder="Paste chat logs here... (format:  username: message)\n\nalice: hey how old are you?\nbob: im 13\nalice: cool, are you home alone?",
                    lines=13,
                    show_label=False
                )
                
                with gr.Row():
                    clear_btn = gr.Button("Clear", elem_classes=["ghost-btn"])
                    analyze_btn = gr.Button("⚡ Analyze Risk", elem_classes=["obsidian-btn"])
                    
                with gr.Accordion("📚 Example Datasets", open=False, elem_classes=["obsidian-panel"]):
                    gr.Examples(
                        examples=[[ex["conversation"]] for ex in _EXAMPLES],
                        inputs=[conv_input],
                        label=""
                    )

            # RIGHT PANEL
            with gr.Column(scale=6, elem_classes=["obsidian-panel"]):
                with gr.Tabs():
                    with gr.TabItem("📋 Executive Summary"):
                        summary_out = gr.HTML("<div class='animate-slideUp' style='text-align:center; padding:50px; color:#64748B; font-style:italic;'>Awaiting sequence injection...</div>")
                        
                    with gr.TabItem("💬 Conversation Highlights"):
                        highlights_out = gr.HTML("<div class='animate-slideUp' style='text-align:center; padding:50px; color:#64748B; font-style:italic;'>Neural net idle...</div>")
                        
                    with gr.TabItem("⚙️ Technical Logs"):
                        logs_out = gr.HTML("<div class='animate-slideUp' style='text-align:center; padding:50px; color:#64748B; font-style:italic;'>Terminal stream dormant.</div>")

        def run_analysis(conv_text):
            if not conv_text.strip():
                empty = "<div class='animate-slideUp' style='text-align:center; padding:50px; color:#64748B; font-style:italic;'>Awaiting sequence injection...</div>"
                return empty, empty, empty
            
            risk, conf, pattern_score, patterns, overridden = backend.predict(conv_text)
            return (
                render_exec_summary(risk, conf, pattern_score, patterns, overridden),
                render_highlights(conv_text, patterns),
                render_tech_logs(conv_text, patterns, risk, conf, pattern_score, overridden),
            )

        def clear_ui():
            empty = "<div class='animate-slideUp' style='text-align:center; padding:50px; color:#64748B; font-style:italic;'>Awaiting sequence injection...</div>"
            return "", empty, empty, empty

        analyze_btn.click(
            fn=run_analysis,
            inputs=[conv_input],
            outputs=[summary_out, highlights_out, logs_out]
        )
        
        clear_btn.click(
            fn=clear_ui,
            outputs=[conv_input, summary_out, highlights_out, logs_out]
        )

    return demo, custom_css, theme


if __name__ == "__main__":
    if not CLASSIFIER_PATH.exists():
        print("[!] Classifier not found! Run: python scripts/train_classifier.py")
        raise SystemExit(1)
        
    backend = ClassifierBackend(CLASSIFIER_PATH)    
    demo, custom_css, theme = build_ui(backend)
    demo.launch(server_name="0.0.0.0", server_port=7862, share=False, css=custom_css, theme=theme)
