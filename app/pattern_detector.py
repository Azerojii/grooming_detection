"""
pattern_detector.py
Rule-based grooming indicator detection for explainability.

Returns structured PatternMatch objects referencing specific conversation lines.
"""

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Pattern categories
# ---------------------------------------------------------------------------

@dataclass
class PatternMatch:
    category: str
    line: str
    explanation: str


PATTERN_WEIGHTS: dict[str, float] = {
    "Age Probing": 0.15,
    "Isolation Check": 0.35,
    "Secrecy Enforcement": 0.45,
    "Photo/Webcam Solicitation": 0.45,
    "Personal Info Solicitation": 0.25,
    "Meeting Request": 0.55,
    "Sexual Content": 0.7,
    "Flattery/Trust Building": 0.2,
}

_PATTERNS: dict[str, list[tuple[re.Pattern, str]]] = {
    "Age Probing": [
        (re.compile(r"\b(how old|what age|whats? ur age|a/?s/?l)\b", re.I),
         "Soliciting age information"),
        (re.compile(r"\b(are [uy]ou?|r u|ur)\s+\d{1,2}\b", re.I),
         "Age-related questioning"),
        (re.compile(r"\b(i[' ]?m|im|i am)\s+\d{1,2}\b", re.I),
         "Age disclosure (may indicate probing response)"),
    ],
    "Isolation Check": [
        (re.compile(r"\bhome\s*alone\b", re.I),
         "Checking if target is unsupervised"),
        (re.compile(r"\bparents?\s*(away|gone|at work|out|not home|asleep|sleeping)\b", re.I),
         "Probing parental absence"),
        (re.compile(r"\b(anyone|anybody|someone)\s*(home|there|around)\b", re.I),
         "Checking for supervision"),
    ],
    "Secrecy Enforcement": [
        (re.compile(r"\b(don[' ]?t|do not|dont)\s+tell\b", re.I),
         "Requesting secrecy"),
        (re.compile(r"\b(our|little|my)\s+secret\b", re.I),
         "Framing interaction as secret"),
        (re.compile(r"\bbetween\s+(us|you and me|u and me)\b", re.I),
         "Enforcing secrecy between parties"),
        (re.compile(r"\b(no\s*one|nobody)\s*(needs? to|has to|should)\s*know\b", re.I),
         "Enforcing secrecy"),
    ],
    "Photo/Webcam Solicitation": [
        (re.compile(r"\b(send|show)\s*(me\s*)?(a\s*)?(pic|pix|photo|selfie)\b", re.I),
         "Soliciting photos"),
        (re.compile(r"\b(web\s*cam|cam2cam|cam\s+to\s+cam|turn\s*(on|ur)\s*cam)\b", re.I),
         "Requesting webcam access"),
        (re.compile(r"\bvideo\s*call\b", re.I),
         "Requesting video interaction"),
    ],
    "Personal Info Solicitation": [
        (re.compile(r"\b(where\s*(do|r)?\s*(you|u|ya)\s*(live|stay)|what\s*city|what\s*town|what\s*state)\b", re.I),
         "Soliciting location information"),
        (re.compile(r"\b(phone|cell|mobile)\s*(number|#|num)\b", re.I),
         "Soliciting phone number"),
        (re.compile(r"\b(what\s*school|which\s*school|where.*go\s*to\s*school)\b", re.I),
         "Soliciting school information"),
    ],
    "Meeting Request": [
        (re.compile(r"\b(wanna|want\s*to|lets?|let[' ]?s)\s*(meet|hang|hook\s*up|get\s*together|come\s*over)\b", re.I),
         "Requesting in-person meeting"),
        (re.compile(r"\b(come\s*(to|over)|pick\s*(you|u)\s*up|visit\s*(you|u))\b", re.I),
         "Proposing physical meeting"),
        (re.compile(r"\b(meet\s*(up|me|irl)|in\s*person)\b", re.I),
         "Requesting real-life meeting"),
    ],
    "Sexual Content": [
        (re.compile(r"\b(horny|sexy|naked|nude|undress|take\s*(off|it\s*off)|strip)\b", re.I),
         "Sexually explicit language"),
        (re.compile(r"\b(wanna\s*f[u*]ck|have\s*sex|sleep\s*with\s*(me|you|u)|make\s*love)\b", re.I),
         "Sexual solicitation"),
        (re.compile(r"\b(suck|lick|touch\s*(yourself|urself|my|ur))\b", re.I),
         "Sexually explicit content"),
        (re.compile(r"\b(dick|cock|pussy|tits|boobs|penis|vagina)\b", re.I),
         "Explicit anatomical references"),
    ],
    "Flattery/Trust Building": [
        (re.compile(r"\b(you[' ]?re|ur|u r)\s*(so\s*)?(mature|special|different|smart|beautiful|pretty|hot|cute|sexy)\b", re.I),
         "Excessive flattery targeting minor"),
        (re.compile(r"\b(nobody|no\s*one)\s*(understands?|gets?|knows?)\s*(you|u)\s*(like|the\s*way)\b", re.I),
         "Isolation through false intimacy"),
        (re.compile(r"\b(you can trust me|trust me|i[' ]?m?\s*not\s*like\s*(other|the\s*rest))\b", re.I),
         "Trust-building manipulation"),
    ],
}


def detect_patterns(conversation: str) -> list[PatternMatch]:
    """Scan each line of the conversation for grooming indicators."""
    matches = []
    for line in conversation.strip().splitlines():
        line_stripped = line.strip()
        if not line_stripped:
            continue
        for category, patterns in _PATTERNS.items():
            for regex, explanation in patterns:
                if regex.search(line_stripped):
                    matches.append(PatternMatch(
                        category=category,
                        line=line_stripped,
                        explanation=explanation,
                    ))
                    break  # one match per category per line
    return matches
