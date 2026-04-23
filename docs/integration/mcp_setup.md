# MCP Server Setup Guide

## Overview

GrooveGenius exposes a Model Context Protocol (MCP) server so external AI agents can programmatically discover and call the recommender as a tool.

**Entry point:** `python -m src.mcp_server`

## Available Tools

| Tool | Requires LLM | Description |
|---|---|---|
| `recommend` | Yes (Ollama) | Natural language query → agent loop → ranked results |
| `recommend_manual` | No | Structured preferences → direct scoring → ranked results |
| `explain_song` | No | Score breakdown for a specific song against a profile |
| `audit_bias` | No | Run automated bias detection across synthetic profiles |
| `list_catalog` | No | Browse songs, optionally filtered by genre or mood |

## Available Resources (read-only)

| URI | Description |
|---|---|
| `groovegenius://catalog/songs` | Full catalog as JSON |
| `groovegenius://catalog/stats` | Genre/mood counts, energy distribution |
| `groovegenius://strategies` | 4 ranking strategies with weight configs |
| `groovegenius://audit/latest` | Most recent bias audit report |

## Client Configuration

### Claude Code

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "groovegenius": {
      "command": "python",
      "args": ["-m", "src.mcp_server"],
      "cwd": "/absolute/path/to/music-recommender"
    }
  }
}
```

### Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "groovegenius": {
      "command": "python",
      "args": ["-m", "src.mcp_server"],
      "cwd": "/absolute/path/to/music-recommender",
      "env": {
        "PYTHONPATH": "/absolute/path/to/music-recommender"
      }
    }
  }
}
```

### Custom Python Agent

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def use_recommender():
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "src.mcp_server"],
        cwd="/absolute/path/to/music-recommender",
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # List available tools
            tools = await session.list_tools()
            print([t.name for t in tools.tools])

            # Get recommendations (no LLM required)
            result = await session.call_tool("recommend_manual", {
                "genre": "lofi",
                "mood": "chill",
                "energy": 0.3,
                "likes_acoustic": True,
                "k": 5,
            })
            print(result)

            # Explain a specific song
            result = await session.call_tool("explain_song", {
                "song_title": "Midnight Coding",
                "genre": "lofi",
                "mood": "chill",
                "energy": 0.3,
                "likes_acoustic": True,
            })
            print(result)
```
