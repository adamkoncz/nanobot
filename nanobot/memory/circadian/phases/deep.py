import json
import re
from typing import TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from nanobot.memory.circadian.plugin import CircadianMemoryPlugin
    from nanobot.providers.base import LLMProvider


class DeepSleepPhase:
    """
    Deep Sleep: Durable promotion and synaptic pruning.
    Processes 'inbox' nodes into atomic knowledge graph nodes with Wikilinks.
    """

    def __init__(self, plugin: "CircadianMemoryPlugin"):
        self.plugin = plugin
        self.vault = plugin.vault
        self.index = plugin.index

    async def run(self) -> int:
        provider = self.plugin._provider
        model = self.plugin._model
        
        if not provider:
            logger.warning("DeepSleep: No LLM provider available. Skipping.")
            return 0

        nodes = self.vault.list_nodes()
        staging_nodes = []
        for n in nodes:
            try:
                node = self.vault.read_node(n)
                if node.metadata.get("type") == "staging" and node.metadata.get("status") == "unprocessed":
                    staging_nodes.append(node)
            except Exception:
                continue

        if not staging_nodes:
            logger.debug("DeepSleep: No staging nodes to process.")
            return 0

        logger.info(f"DeepSleep: Processing {len(staging_nodes)} staging nodes.")
        
        processed_count = 0
        for staging_node in staging_nodes:
            await self._process_staging_node(staging_node, provider, model)
            self.vault.delete_node(staging_node.path.stem)
            self.index.remove_node(staging_node.path.stem)
            processed_count += 1
            
        self._rebuild_root_index()
        return processed_count

    async def _process_staging_node(self, node, provider: "LLMProvider", model: str):
        prompt = f"""
You are the Deep Sleep processor for a Circadian Knowledge Graph.
Your task is to take the following staging notes and extract atomic concepts, people, and topics.
Output a JSON list of objects, each representing a node to create or update.
Use Wikilinks `[[ConceptName]]` in the content to link them.

Staging Note:
{node.content}

Return ONLY valid JSON matching this schema:
[
  {{
    "name": "ConceptName",
    "content": "Description of the concept...",
    "connections": ["OtherConcept1", "OtherConcept2"]
  }}
]
"""
        response = await provider.chat(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.1
        )
        
        try:
            content = response.content or "[]"
            match = re.search(r'```(?:json)?\s*(\[.*\])\s*```', content, re.DOTALL)
            if match:
                content = match.group(1)
            
            extracted_nodes = json.loads(content)
            
            for en in extracted_nodes:
                name = en.get("name", "").strip().replace(" ", "_").lower()
                if not name:
                    continue
                
                new_content = en.get("content", "")
                connections = en.get("connections", [])
                
                try:
                    existing = self.vault.read_node(name)
                    updated_content = existing.content + "\n\n" + new_content
                    self.vault.write_node(name, updated_content, existing.metadata)
                    self.index.index_node(name, updated_content)
                except FileNotFoundError:
                    metadata = {"type": "concept", "connections": connections}
                    self.vault.write_node(name, new_content, metadata)
                    self.index.index_node(name, new_content)

        except Exception as e:
            logger.error(f"DeepSleep: Failed to process staging node {node.path.stem}: {e}")

    def _rebuild_root_index(self):
        """Rebuilds the root index.md which acts as the entrypoint for the LLM."""
        nodes = self.vault.list_nodes()
        index_content = "# Knowledge Graph Index\n\n"
        
        concept_nodes = []
        for n in nodes:
            if n == "index" or n.startswith("inbox_"):
                continue
            concept_nodes.append(n)
            
        for n in sorted(concept_nodes):
            index_content += f"- [[{n}]]\n"
            
        self.vault.write_node("index", index_content, {"type": "index"})
        self.index.index_node("index", index_content)
