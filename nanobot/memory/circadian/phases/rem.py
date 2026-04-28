import random
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from nanobot.memory.circadian.plugin import CircadianMemoryPlugin


class REMSleepPhase:
    """
    REM Sleep: Creative synthesis.
    Randomly samples nodes to find hidden connections and generate new insights.
    """

    def __init__(self, plugin: "CircadianMemoryPlugin"):
        self.plugin = plugin
        self.vault = plugin.vault
        self.index = plugin.index

    async def run(self) -> int:
        provider = self.plugin._provider
        model = self.plugin._model
        
        if not provider:
            logger.warning("REMSleep: No LLM provider available. Skipping.")
            return 0

        nodes = self.vault.list_nodes()
        concept_nodes = [n for n in nodes if not n.startswith("inbox_") and n != "index"]
        
        if len(concept_nodes) < 2:
            logger.debug("REMSleep: Not enough concept nodes to synthesize.")
            return 0

        sample_size = min(random.randint(3, 5), len(concept_nodes))
        sampled = random.sample(concept_nodes, sample_size)
        
        logger.info(f"REMSleep: Synthesizing concepts: {sampled}")
        
        sampled_content = ""
        for name in sampled:
            node = self.vault.read_node(name)
            sampled_content += f"\n### [[{name}]]\n{node.content}\n"

        prompt = f"""
You are the REM Sleep phase of an AI agent's memory system.
Your goal is to creatively synthesize the following disconnected memories/concepts into a novel insight, pattern, or overarching idea.

Memories:
{sampled_content}

Generate a short, insightful summary of how these concepts connect. 
Provide a "Title" for this new insight and use Wikilinks `[[ConceptName]]` to reference the original concepts.
Format as Markdown with an h1 `# Title` followed by the insight.
"""
        response = await provider.chat(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=1.2
        )
        
        insight = response.content
        if not insight:
            return 0
            
        title_match = re.search(r"^#\s+(.+)$", insight, re.MULTILINE)
        if title_match:
            name = title_match.group(1).strip().replace(" ", "_").lower()
        else:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            name = f"insight_{timestamp}"
            
        content = re.sub(r"^#\s+.+\n+", "", insight, count=1, flags=re.MULTILINE).strip()
            
        metadata = {
            "type": "insight",
            "source_nodes": sampled,
            "created": datetime.now(timezone.utc).isoformat()
        }
        
        self.vault.write_node(name, content, metadata)
        self.index.index_node(name, content)
        
        usage = response.usage
        if usage:
            logger.info(f"REMSleep token usage: {usage}")
            
        return 1
