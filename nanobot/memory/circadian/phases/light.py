from datetime import datetime, timezone
from typing import TYPE_CHECKING

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
        model = self.plugin._model
        
        if not provider:
            logger.warning("LightSleep: No LLM provider available. Staging raw text.")
            staged_content = self._stage_raw(unprocessed)
        else:
            staged_content = await self._stage_with_llm(unprocessed, provider, model)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        node_name = f"inbox_{timestamp}"
        
        metadata = {
            "type": "staging",
            "status": "unprocessed",
            "created": datetime.now(timezone.utc).isoformat(),
            "source_records": len(unprocessed)
        }
        
        self.vault.write_node(node_name, staged_content, metadata)
        self.index.index_node(node_name, staged_content)
        
        self.plugin.set_last_dream_cursor(cursor + len(unprocessed))
        return len(unprocessed)

    def _stage_raw(self, entries: list[dict]) -> str:
        lines = []
        for e in entries:
            role = e.get("role", "unknown")
            content = e.get("content", "")
            if isinstance(content, list):
                content = " ".join([c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"])
            lines.append(f"**{role}**: {content}")
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
        response = await provider.chat(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.3
        )
        return response.content or raw_text
