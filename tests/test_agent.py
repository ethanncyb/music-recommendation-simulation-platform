"""Tests for agent tools, conversation state, and logger.

These tests use fallback logic and mock LLM responses so they run
without Ollama or any LLM backend.
"""

import json
import os
import tempfile

from src.agent_tools import (
    extract_profile, select_strategy, critique_results, adjust_weights,
    _fallback_profile, _validate_profile, _fallback_adjust,
)
from src.agent_logger import AgentLogger
from src.conversation import ConversationState
from src.recommender import DEFAULT, GENRE_FIRST, RankingStrategy, load_songs
from src.agent import AgentLoop


# ── Mock LLM that returns pre-set JSON ────────────────────────────────────

class MockLLM:
    """A mock LLM that returns a fixed response."""
    def __init__(self, response: str = "{}"):
        self.response = response
        self.calls = []

    def generate(self, prompt, system=None):
        self.calls.append(prompt)
        return self.response

    def generate_json(self, prompt, system=None):
        self.calls.append(prompt)
        return json.loads(self.response)


class FailingLLM:
    """A mock LLM that always raises."""
    def generate(self, prompt, system=None):
        raise ConnectionError("LLM unavailable")

    def generate_json(self, prompt, system=None):
        raise ConnectionError("LLM unavailable")


# ── Shared test data ─────────────────────────────────────────────────────

VALID_GENRES = ["ambient", "classical", "edm", "lofi", "metal", "pop", "rock", "synthwave"]
VALID_MOODS = ["angry", "chill", "energetic", "happy", "intense", "moody", "relaxed", "sad"]


# ── extract_profile tests ────────────────────────────────────────────────

def test_extract_profile_with_mock_llm():
    llm = MockLLM(json.dumps({
        "genre": "synthwave", "mood": "moody", "energy": 0.75,
        "likes_acoustic": False, "preferred_tags": ["driving", "retro"],
    }))
    profile = extract_profile("night drive music", llm, VALID_GENRES, VALID_MOODS)
    assert profile["genre"] == "synthwave"
    assert profile["mood"] == "moody"
    assert profile["energy"] == 0.75
    assert profile["likes_acoustic"] is False
    assert "driving" in profile["preferred_tags"]


def test_extract_profile_validates_invalid_genre():
    llm = MockLLM(json.dumps({
        "genre": "lo-fi", "mood": "chill", "energy": 0.3,
        "likes_acoustic": True,
    }))
    profile = extract_profile("chill study music", llm, VALID_GENRES, VALID_MOODS)
    # "lo-fi" should be snapped to "lofi" via substring match
    assert profile["genre"] == "lofi"


def test_extract_profile_clamps_energy():
    llm = MockLLM(json.dumps({
        "genre": "pop", "mood": "happy", "energy": 1.5,
        "likes_acoustic": False,
    }))
    profile = extract_profile("happy music", llm, VALID_GENRES, VALID_MOODS)
    assert profile["energy"] == 1.0  # clamped


def test_extract_profile_fallback_on_failure():
    llm = FailingLLM()
    profile = extract_profile("chill lofi study beats", llm, VALID_GENRES, VALID_MOODS)
    # Fallback should pick up "lofi" and "chill" from keywords
    assert profile["genre"] == "lofi"
    assert profile["mood"] == "chill"


def test_fallback_profile_keyword_detection():
    profile = _fallback_profile("intense metal workout", VALID_GENRES, VALID_MOODS)
    assert profile["genre"] == "metal"
    assert profile["mood"] == "intense"
    assert profile["energy"] == 0.8  # "workout" = high energy


def test_fallback_profile_acoustic_detection():
    profile = _fallback_profile("acoustic guitar folk", VALID_GENRES, VALID_MOODS)
    assert profile["likes_acoustic"] is True


# ── select_strategy tests ────────────────────────────────────────────────

def test_select_strategy_with_mock():
    llm = MockLLM(json.dumps({"strategy": "mood_first"}))
    profile = {"genre": "lofi", "mood": "chill", "energy": 0.3, "likes_acoustic": True}
    result = select_strategy(profile, llm)
    assert result.name == "Mood-First"


def test_select_strategy_fallback():
    llm = FailingLLM()
    profile = {"genre": "pop", "mood": "happy", "energy": 0.8}
    result = select_strategy(profile, llm)
    assert result.name == "Default"  # fallback


# ── critique_results tests ───────────────────────────────────────────────

def test_critique_approved():
    llm = MockLLM(json.dumps({
        "approved": True, "issues": [], "adjustments": {},
    }))
    results = [({"title": "Song", "genre": "pop", "mood": "happy", "energy": 0.8}, 0.9, "match")]
    critique = critique_results("happy pop", {"genre": "pop"}, results, llm)
    assert critique["approved"] is True


def test_critique_not_approved():
    llm = MockLLM(json.dumps({
        "approved": False,
        "issues": ["wrong genre"],
        "adjustments": {"genre": "rock"},
    }))
    results = [({"title": "Song", "genre": "pop", "mood": "happy", "energy": 0.8}, 0.5, "weak")]
    critique = critique_results("rock music", {"genre": "pop"}, results, llm)
    assert critique["approved"] is False
    assert "wrong genre" in critique["issues"]
    assert critique["adjustments"]["genre"] == "rock"


def test_critique_fallback():
    llm = FailingLLM()
    results = [({"title": "Song", "genre": "pop", "mood": "happy", "energy": 0.8}, 0.9, "match")]
    critique = critique_results("happy pop", {"genre": "pop"}, results, llm)
    assert critique["approved"] is True  # fallback approves


# ── adjust_weights tests ─────────────────────────────────────────────────

def test_adjust_weights_with_mock():
    llm = MockLLM(json.dumps({
        "genre": 0.10, "mood": 0.30, "energy": 0.40, "acoustic": 0.20,
    }))
    result = adjust_weights("more acoustic", DEFAULT, llm)
    assert isinstance(result, RankingStrategy)
    assert abs(result.genre_weight + result.mood_weight +
               result.energy_weight + result.acoustic_weight - 1.0) < 0.01


def test_adjust_weights_fallback():
    result = _fallback_adjust("less electronic, more acoustic", DEFAULT)
    assert result.acoustic_weight > DEFAULT.acoustic_weight


def test_adjust_weights_normalizes():
    llm = MockLLM(json.dumps({
        "genre": 0.5, "mood": 0.5, "energy": 0.5, "acoustic": 0.5,
    }))
    result = adjust_weights("balance", DEFAULT, llm)
    total = result.genre_weight + result.mood_weight + result.energy_weight + result.acoustic_weight
    assert abs(total - 1.0) < 0.01


# ── ConversationState tests ──────────────────────────────────────────────

def test_conversation_add_turn():
    state = ConversationState()
    state.add_turn("user", "I want chill music")
    state.add_turn("agent", "Here are your recommendations")
    assert len(state.turn_history) == 2
    assert state.turn_history[0]["role"] == "user"


def test_conversation_apply_adjustments():
    state = ConversationState()
    state.current_profile = {"genre": "pop", "mood": "happy", "energy": 0.8, "likes_acoustic": False}
    state.apply_adjustments({"genre": "rock", "energy": 0.5})
    assert state.current_profile["genre"] == "rock"
    assert state.current_profile["energy"] == 0.5
    assert state.refinement_count == 1


def test_conversation_context_summary():
    state = ConversationState()
    state.current_profile = {"genre": "lofi", "mood": "chill", "energy": 0.3, "likes_acoustic": True}
    state.add_turn("user", "chill study music")
    summary = state.get_context_summary()
    assert "lofi" in summary
    assert "chill" in summary


# ── AgentLogger tests ─────────────────────────────────────────────────────

def test_logger_records_steps():
    logger = AgentLogger(session_id="test-session")
    logger.log_step("plan", input_data={"query": "test"}, output_data={"genre": "pop"})
    logger.log_step("execute", input_data={}, output_data={"count": 5}, duration_ms=10)
    assert len(logger.steps) == 2
    assert logger.steps[0].step == "plan"


def test_logger_saves_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = AgentLogger(session_id="test-save")
        logger.log_step("plan", input_data={"q": "test"}, output_data={"g": "pop"})
        path = logger.save(output_dir=tmpdir)
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert data["session_id"] == "test-save"
        assert data["step_count"] == 1


# ── AgentLoop integration test (with mock LLM) ───────────────────────────

def test_agent_loop_run_with_mock():
    """Full agent loop with mock LLM — tests the orchestration without real LLM."""
    songs = load_songs("data/songs.csv")

    # Mock that returns different responses based on call order
    class SequenceLLM:
        def __init__(self):
            self.call_count = 0
            self.responses = [
                # extract_profile
                json.dumps({"genre": "pop", "mood": "happy", "energy": 0.85,
                            "likes_acoustic": False, "preferred_tags": ["upbeat"]}),
                # select_strategy
                json.dumps({"strategy": "default"}),
                # critique_results
                json.dumps({"approved": True, "issues": [], "adjustments": {}}),
            ]

        def generate(self, prompt, system=None):
            return self.responses[min(self.call_count, len(self.responses) - 1)]

        def generate_json(self, prompt, system=None):
            resp = self.responses[min(self.call_count, len(self.responses) - 1)]
            self.call_count += 1
            return json.loads(resp)

    llm = SequenceLLM()
    agent = AgentLoop(llm=llm, songs=songs)
    result = agent.run("upbeat pop music for a party")

    assert "results" in result
    assert len(result["results"]) > 0
    assert result["profile"]["genre"] == "pop"
    assert result["strategy"] == "Default"
    assert "confidence" in result
    assert "reasoning_trace" in result
    assert len(result["reasoning_trace"]) >= 2  # at least PLAN + EXECUTE
