import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import uuid4

from loguru import logger

if TYPE_CHECKING:
    from nanobot.memory.circadian.plugin import CircadianMemoryPlugin
    from nanobot.providers.base import LLMProvider


class LightSleepPhase:
    """
    Light Sleep: acts as a librarian organizing the previous day's data.
    Takes unprocessed history.jsonl and stages it in the Vault Inbox.
    """

    def __init__(self, plugin: "CircadianMemoryPlugin"):
        self.plugin = plugin
        self.vault = plugin.vault
        self.index = plugin.index

    async def run(self) -> int:
        """
        Runs the Light Sleep phase.
        Returns the number of processed records.
        """
        cursor = self.plugin.get_last_dream_cursor()
        unprocessed = self.plugin.read_unprocessed_history(cursor)

        if not unprocessed:
            logger.debug("LightSleep: No new history to process.")
            return 0

        logger.info(f"LightSleep: Processing {len(unprocessed)} new history entries.")

        provider = self.plugin._provider
        model = self.plugin._model_override or self.plugin._model
        
        if not provider:
            logger.warning("LightSleep: No LLM provider available. Staging raw text.")
            staged_content = self._stage_raw(unprocessed)
        else:
            staged_content = await self._stage_with_llm(unprocessed, provider, model)

        # Use ISO timestamp + short UUID to avoid collisions
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        unique_suffix = uuid4().hex[:6]
        node_name = f"inbox_{timestamp}_{unique_suffix}"
        
        metadata = {
            "type": "staging",
            "status": "unprocessed",
            "created": datetime.now(timezone.utc).isoformat(),
            "source_records": len(unprocessed)
        }
        
        self.vault.write_node(node_name, staged_content, metadata)
        self.index.index_node(node_name, staged_content, node_type="staging")
        
        # Use the max actual cursor value from entries, not len()
        max_cursor = max(
            (e.get("cursor", 0) for e in unprocessed),
            default=cursor
        )
        self.plugin.set_last_dream_cursor(max_cursor)
        return len(unprocessed)

    def _stage_raw(self, entries: list[dict]) -> str:
        """Format history entries as readable Markdown, including tool calls."""
        lines = []
        for e in entries:
            role = e.get("role", "unknown")
            content = e.get("content", "")

            # Handle multimodal content blocks
            if isinstance(content, list):
                parts = []
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        parts.append(c.get("text", ""))
                content = " ".join(parts)

            # Handle tool call messages
            tool_calls = e.get("tool_calls")
            tools_used = e.get("tools_used")
            tool_suffix = ""
            if tool_calls and isinstance(tool_calls, list):
                tool_names = []
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "") if isinstance(fn, dict) else ""
                    if name:
                        tool_names.append(name)
                if tool_names:
                    tool_suffix = f" [tools: {', '.join(tool_names)}]"
            elif tools_used and isinstance(tools_used, list):
                tool_suffix = f" [tools: {', '.join(tools_used)}]"

            # Handle tool response messages
            if role == "tool":
                tool_name = e.get("name", "unknown_tool")
                content_preview = (str(content)[:200] + "...") if len(str(content)) > 200 else str(content)
                lines.append(f"**tool** ({tool_name}): {content_preview}")
                continue

            if content:
                lines.append(f"**{role}**{tool_suffix}: {content}")
            elif tool_suffix:
                lines.append(f"**{role}**{tool_suffix}")

        return "\n\n".join(lines)

    async def _stage_with_llm(self, entries: list[dict], provider: "LLMProvider", model: str) -> str:
        raw_text = self._stage_raw(entries)
        
        prompt = f"""
You are a librarian in the Light Sleep phase of memory consolidation.
Review the following recent interaction history and summarize the key facts, tasks, and context into a clear, structured Markdown document.

Interaction History:
{raw_text}

Provide the summary as clean Markdown.
"""
        response = await provider.chat_with_retry(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.3
        )

        if response.finish_reason == "error":
            logger.warning(f"LightSleep: LLM call failed ({response.content}), staging raw text.")
            return raw_text

        return response.content or raw_text
