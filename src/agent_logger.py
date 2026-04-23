"""
Structured logging for agent sessions.

Logs every step (plan, execute, critique, refine, respond) with input/output
data, LLM reasoning, errors, and timing. Saves as JSON for later review.

Usage:
    from src.agent_logger import AgentLogger

    logger = AgentLogger()
    logger.log_step("plan", input_data={...}, output_data={...}, duration_ms=42)
    logger.save()  # writes to logs/agent_runs/<session_id>.json
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class AgentLog:
    """A single step in an agent session."""
    timestamp: str
    step: str
    input_data: Dict
    output_data: Dict
    llm_reasoning: str = ""
    errors: List[str] = field(default_factory=list)
    duration_ms: int = 0


class AgentLogger:
    """Logs and persists agent session steps."""

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.steps: List[AgentLog] = []
        self.start_time = time.time()

    def log_step(self, step: str, input_data: Dict = None,
                 output_data: Dict = None, llm_reasoning: str = "",
                 errors: List[str] = None, duration_ms: int = 0):
        """Record a single agent step."""
        self.steps.append(AgentLog(
            timestamp=datetime.now().isoformat(),
            step=step,
            input_data=input_data or {},
            output_data=output_data or {},
            llm_reasoning=llm_reasoning,
            errors=errors or [],
            duration_ms=duration_ms,
        ))

    def save(self, output_dir: str = "logs/agent_runs") -> str:
        """Save the full session log as JSON. Returns the file path."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{self.session_id}.json")

        data = {
            "session_id": self.session_id,
            "total_duration_ms": int((time.time() - self.start_time) * 1000),
            "step_count": len(self.steps),
            "steps": [asdict(s) for s in self.steps],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        return path
