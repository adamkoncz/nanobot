import random
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import uuid4

from loguru import logger

from nanobot.memory.circadian.vault import sanitize_node_name

if TYPE_CHECKING:
    from nanobot.memory.circadian.plugin import CircadianMemoryPlugin


class REMSleepPhase:
    """
    REM Sleep: Creative synthesis.
    Randomly samples nodes to find hidden connections and generate new insights.
    """

    def __init__(self, plugin: "CircadianMemoryPlugin", *, token_limit: int = 10_000):
        self.plugin = plugin
        self.vault = plugin.vault
        self.index = plugin.index
        self.token_limit = token_limit

    async def run(self) -> int:
        provider = self.plugin._provider
        model = self.plugin._model_override or self.plugin._model
        
        if not provider:
            logger.warning("REMSleep: No LLM provider available. Skipping.")
            return 0

        nodes = self.vault.list_nodes()
        concept_nodes = [n for n in nodes if not n.startswith("inbox_") and n != "index" and not n.startswith("insight_")]
        
        if len(concept_nodes) < 2:
            logger.debug("REMSleep: Not enough concept nodes to synthesize.")
            return 0

        sample_size = min(random.randint(3, 5), len(concept_nodes))
        sampled = random.sample(concept_nodes, sample_size)
        
        logger.info(f"REMSleep: Synthesizing concepts: {sampled}")
        
        sampled_content = ""
        for name in sampled:
            try:
                node = self.vault.read_node(name)
                sampled_content += f"\n### [[{name}]]\n{node.content}\n"
            except FileNotFoundError:
                continue

        if not sampled_content.strip():
            return 0

        prompt = f"""
You are the REM Sleep phase of an AI agent's memory system.
Your goal is to creatively synthesize the following disconnected memories/concepts into a novel insight, pattern, or overarching idea.

Memories:
{sampled_content}

Generate a short, insightful summary of how these concepts connect. 
Provide a "Title" for this new insight and use Wikilinks `[[ConceptName]]` to reference the original concepts.
Format as Markdown with an h1 `# Title` followed by the insight.
"""
        response = await provider.chat_with_retry(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=1.2
        )

        if response.finish_reason == "error":
            logger.warning(f"REMSleep: LLM call failed: {response.content}")
            return 0
        
        insight = response.content
        if not insight:
            return 0

        # Log token usage
        usage = response.usage
        if usage:
            total_tokens = usage.get("total_tokens", 0)
            logger.info(f"REMSleep token usage: {usage}")
            if total_tokens > self.token_limit:
                logger.warning(
                    f"REMSleep: Token usage ({total_tokens}) exceeded limit ({self.token_limit})."
                )

        # Build a collision-safe insight name with prefix
        title_match = re.search(r"^#\s+(.+)$", insight, re.MULTILINE)
        if title_match:
            slug = sanitize_node_name(title_match.group(1))
        else:
            slug = ""

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        unique = uuid4().hex[:6]
        if slug:
            name = f"insight_{slug}_{unique}"
        else:
            name = f"insight_{timestamp}_{unique}"
            
        content = re.sub(r"^#\s+.+\n+", "", insight, count=1, flags=re.MULTILINE).strip()
            
        metadata = {
            "type": "insight",
            "source_nodes": sampled,
            "created": datetime.now(timezone.utc).isoformat()
        }
        
        self.vault.write_node(name, content, metadata)
        self.index.index_node(name, content, node_type="insight")
            
        return 1
