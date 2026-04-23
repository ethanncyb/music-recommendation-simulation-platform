"""
Agentic recommendation loop.

Orchestrates the plan → execute → critique → refine → respond cycle
using LLM-powered tool functions and structured logging.

Usage:
    from src.agent import AgentLoop
    from src.llm_provider import get_provider

    llm = get_provider("ollama")
    agent = AgentLoop(llm=llm, songs=songs, knowledge=knowledge)

    # Single-turn
    result = agent.run("moody driving music for a night road trip")

    # Multi-turn
    result = agent.chat("moody driving music")
    result = agent.chat("less electronic, more acoustic")
"""

import time
from typing import Dict, List, Optional

from .recommender import (
    load_songs, recommend_songs, RankingStrategy, DEFAULT,
)
from .llm_provider import LLMProvider
from .agent_tools import extract_profile, select_strategy, critique_results, adjust_weights
from .agent_logger import AgentLogger
from .conversation import ConversationState
from .confidence import ConfidenceScorer
from .guardrails import apply_guardrails, format_confidence_badge
from .self_critique import self_critique, self_critique_offline


MAX_REFINEMENTS = 2


class AgentLoop:
    """LLM-powered agentic recommendation loop."""

    def __init__(self, llm: LLMProvider, songs: List[Dict],
                 knowledge: Optional[Dict] = None):
        self.llm = llm
        self.songs = songs
        self.knowledge = knowledge
        self.valid_genres = sorted(set(s["genre"] for s in songs))
        self.valid_moods = sorted(set(s["mood"] for s in songs))
        self.confidence_scorer = ConfidenceScorer(songs)
        self.state = ConversationState()
        self.logger = AgentLogger()

    def run(self, query: str, k: int = 5,
            strategy: Optional[RankingStrategy] = None) -> Dict:
        """Single-turn recommendation from a natural language query.

        Returns a dict with results, profile, strategy, confidence,
        critique, guardrail message, and reasoning trace.
        """
        trace = []

        # ── PLAN ──────────────────────────────────────────────────────
        t0 = time.time()
        profile = extract_profile(query, self.llm, self.valid_genres, self.valid_moods)
        self.state.current_profile = profile

        if strategy is None:
            strategy = select_strategy(profile, self.llm)
        self.state.current_strategy = strategy

        plan_ms = int((time.time() - t0) * 1000)
        trace.append(f"PLAN: Extracted profile ({profile['genre']}/{profile['mood']}, "
                     f"energy={profile['energy']}) → Strategy: {strategy.name}")
        self.logger.log_step("plan",
            input_data={"query": query},
            output_data={"profile": profile, "strategy": strategy.name},
            duration_ms=plan_ms,
        )

        # ── EXECUTE + CRITIQUE loop (max refinements) ─────────────────
        results = None
        confidence = None
        refinement = 0

        while refinement <= MAX_REFINEMENTS:
            # EXECUTE
            t0 = time.time()
            results = recommend_songs(
                profile, self.songs, k=k,
                strategy=strategy, knowledge=self.knowledge,
            )
            exec_ms = int((time.time() - t0) * 1000)
            trace.append(f"EXECUTE: Got {len(results)} results (top: "
                         f"{results[0][0]['title'] if results else 'none'})")
            self.logger.log_step("execute",
                input_data={"profile": profile, "strategy": strategy.name, "k": k},
                output_data={"result_count": len(results),
                             "top_title": results[0][0]["title"] if results else None},
                duration_ms=exec_ms,
            )

            # CONFIDENCE
            confidence = self.confidence_scorer.compute(profile, results, self.knowledge)

            # CRITIQUE (only if we have refinements left)
            if refinement < MAX_REFINEMENTS:
                t0 = time.time()
                critique_result = critique_results(query, profile, results, self.llm)
                crit_ms = int((time.time() - t0) * 1000)
                self.logger.log_step("critique",
                    input_data={"query": query, "result_count": len(results)},
                    output_data=critique_result,
                    duration_ms=crit_ms,
                )

                if critique_result["approved"]:
                    trace.append("CRITIQUE: Approved — results match intent.")
                    break
                else:
                    # REFINE
                    issues = critique_result.get("issues", [])
                    adjustments = critique_result.get("adjustments", {})
                    trace.append(f"CRITIQUE: Not approved — {'; '.join(issues)}")

                    if adjustments:
                        self.state.apply_adjustments(adjustments)
                        profile = self.state.current_profile
                        trace.append(f"REFINE: Applied adjustments: {adjustments}")
                        self.logger.log_step("refine",
                            input_data={"issues": issues, "adjustments": adjustments},
                            output_data={"new_profile": profile},
                        )
                    else:
                        break  # No adjustments suggested, stop refining

                    refinement += 1
            else:
                break

        # ── RESPOND ───────────────────────────────────────────────────
        guardrail = apply_guardrails(confidence, results)

        critique_text = None
        if guardrail["show_self_critique"]:
            try:
                critique_text = self_critique(
                    query, profile, results, confidence, self.llm,
                )
            except Exception:
                critique_text = self_critique_offline(profile, results, confidence)

        self.state.add_turn("user", query)
        self.state.add_turn("agent", f"Recommended {len(results)} songs")

        self.logger.log_step("respond",
            input_data={"confidence": confidence.overall_confidence},
            output_data={
                "confidence_label": confidence.confidence_label,
                "guardrail": guardrail["guardrail_message"],
                "has_critique": critique_text is not None,
            },
        )

        return {
            "results": results,
            "profile": profile,
            "strategy": strategy.name,
            "confidence": {
                "score": confidence.overall_confidence,
                "label": confidence.confidence_label,
                "warnings": confidence.warnings,
                "signals": confidence.signals,
            },
            "critique": critique_text,
            "guardrail": guardrail["guardrail_message"],
            "reasoning_trace": trace,
        }

    def chat(self, message: str) -> Dict:
        """Multi-turn conversation. First message runs the agent;
        subsequent messages refine the results."""
        if self.state.current_profile is None:
            # First message — do a full run
            return self.run(message)

        # Subsequent message — treat as refinement feedback
        self.state.add_turn("user", message)
        self.state.feedback_log.append(message)

        # Adjust weights based on feedback
        t0 = time.time()
        new_strategy = adjust_weights(
            message, self.state.current_strategy, self.llm,
        )
        self.state.current_strategy = new_strategy
        adj_ms = int((time.time() - t0) * 1000)

        self.logger.log_step("refine",
            input_data={"feedback": message,
                        "old_strategy": self.state.current_strategy.name},
            output_data={"new_strategy": new_strategy.name,
                         "weights": {
                             "genre": new_strategy.genre_weight,
                             "mood": new_strategy.mood_weight,
                             "energy": new_strategy.energy_weight,
                             "acoustic": new_strategy.acoustic_weight,
                         }},
            duration_ms=adj_ms,
        )

        # Re-run with updated strategy
        context = self.state.get_context_summary()
        return self.run(
            f"{context}\nLatest feedback: {message}",
            strategy=new_strategy,
        )

    def save_session(self) -> str:
        """Save the agent session log and return the file path."""
        return self.logger.save()
