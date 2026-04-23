"""
Multi-turn conversation state management for the agentic loop.

Tracks user/agent turns, current profile, strategy, and refinement history
so the agent can handle follow-up messages like "less electronic" or "more variety".
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .recommender import RankingStrategy, DEFAULT


@dataclass
class ConversationState:
    """Tracks the agent's understanding across conversation turns."""
    turn_history: List[Dict] = field(default_factory=list)
    current_profile: Optional[Dict] = None
    current_strategy: RankingStrategy = field(default_factory=lambda: DEFAULT)
    feedback_log: List[str] = field(default_factory=list)
    refinement_count: int = 0

    def add_turn(self, role: str, content: str):
        """Add a conversation turn.

        Args:
            role: "user" or "agent"
            content: message text
        """
        self.turn_history.append({"role": role, "content": content})

    def get_context_summary(self) -> str:
        """Summarize the conversation for LLM context."""
        parts = []

        if self.current_profile:
            p = self.current_profile
            parts.append(
                f"Current preferences: genre={p.get('genre')}, "
                f"mood={p.get('mood')}, energy={p.get('energy')}, "
                f"acoustic={p.get('likes_acoustic')}"
            )

        parts.append(f"Strategy: {self.current_strategy.name}")

        if self.feedback_log:
            parts.append("User feedback: " + "; ".join(self.feedback_log[-3:]))

        # Include last few turns for context
        recent = self.turn_history[-6:]
        if recent:
            parts.append("Recent conversation:")
            for turn in recent:
                parts.append(f"  {turn['role']}: {turn['content'][:100]}")

        return "\n".join(parts)

    def apply_adjustments(self, adjustments: Dict):
        """Apply profile adjustments from a critique or user feedback."""
        if not self.current_profile:
            return

        for key, value in adjustments.items():
            if key in ("genre", "mood") and isinstance(value, str):
                self.current_profile[key] = value.lower().strip()
            elif key == "energy":
                try:
                    self.current_profile["energy"] = max(0.0, min(1.0, float(value)))
                except (ValueError, TypeError):
                    pass
            elif key == "likes_acoustic" and isinstance(value, bool):
                self.current_profile["likes_acoustic"] = value

        self.refinement_count += 1
