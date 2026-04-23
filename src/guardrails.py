"""
Output guardrails based on confidence scoring.

Attaches warnings, caveats, and display hints to recommendation output
based on the confidence level. Does not block results — only adds context.

Usage:
    from src.guardrails import apply_guardrails, format_confidence_badge

    guardrail = apply_guardrails(confidence_report, results)
    badge = format_confidence_badge(confidence_report)
"""

from typing import Dict, List, Optional, Tuple

from .confidence import ConfidenceReport


# Messages by confidence level
_GUARDRAIL_MESSAGES = {
    "high": None,
    "medium": (
        "These are the best matches available, but catalog coverage "
        "is limited for your preferences."
    ),
    "low": (
        "Warning: limited match quality. The catalog may not cover "
        "your preference combination well."
    ),
    "very_low": (
        "Catalog limitation: no strong matches found. The system is returning "
        "the least-bad options. See self-critique for details."
    ),
}

# Badge strings for CLI display
_BADGES = {
    "high": "[HIGH confidence]",
    "medium": "[MEDIUM confidence]",
    "low": "[LOW confidence !!]",
    "very_low": "[VERY LOW confidence !!!]",
}


def apply_guardrails(confidence: ConfidenceReport,
                     results: List[Tuple]) -> Dict:
    """Enrich recommendation output with guardrail information.

    Args:
        confidence: ConfidenceReport from the scorer
        results: recommendation results (unchanged, passed through)

    Returns:
        Dict with:
            results: the original results (unchanged)
            confidence: the ConfidenceReport
            guardrail_message: str or None
            show_self_critique: bool — whether to trigger LLM self-critique
    """
    label = confidence.confidence_label
    message = _GUARDRAIL_MESSAGES.get(label)

    # Append suggestion to message if available
    if message and confidence.suggestion:
        message = f"{message} {confidence.suggestion}"
    elif not message and confidence.suggestion:
        message = confidence.suggestion

    return {
        "results": results,
        "confidence": confidence,
        "guardrail_message": message,
        "show_self_critique": label in ("low", "very_low"),
    }


def format_confidence_badge(confidence: ConfidenceReport) -> str:
    """Return a formatted badge string for CLI display.

    Examples:
        "[HIGH confidence]"
        "[LOW confidence !!]"
    """
    return _BADGES.get(confidence.confidence_label, f"[{confidence.confidence_label}]")
