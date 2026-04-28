import json
import re
from typing import TYPE_CHECKING
from loguru import logger

from nanobot.memory.circadian.vault import sanitize_node_name

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
        model = self.plugin._model_override or self.plugin._model
        
        if not provider:
            logger.warning("DeepSleep: No LLM provider available. Skipping.")
            return 0

        # Use index query instead of full vault scan
        staging_names = self.index.list_by_type("staging")
        if not staging_names:
            logger.debug("DeepSleep: No staging nodes to process.")
            return 0

        logger.info(f"DeepSleep: Processing {len(staging_names)} staging nodes.")
        
        processed_count = 0
        for name in staging_names:
            try:
                node = self.vault.read_node(name)
            except FileNotFoundError:
                # Index out of sync with vault — clean up the orphan
                self.index.remove_node(name)
                continue

            success = await self._process_staging_node(node, provider, model)
            if success:
                # Only delete after successful extraction
                self.vault.delete_node(name)
                self.index.remove_node(name)
                processed_count += 1
            else:
                # Mark as failed so we don't retry indefinitely
                failed_meta = dict(node.metadata)
                failed_meta["status"] = "failed"
                self.vault.write_node(name, node.content, failed_meta)
                self.index.index_node(name, node.content, node_type="staging_failed")
            
        self._rebuild_root_index()
        return processed_count

    async def _process_staging_node(self, node, provider: "LLMProvider", model: str) -> bool:
        """Extract concepts from a staging node. Returns True on success."""
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
        response = await provider.chat_with_retry(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.1
        )
        
        if response.finish_reason == "error":
            logger.warning(f"DeepSleep: LLM call failed for {node.path.stem}: {response.content}")
            return False

        try:
            content = response.content or "[]"
            match = re.search(r'```(?:json)?\s*(\[.*\])\s*```', content, re.DOTALL)
            if match:
                content = match.group(1)
            
            extracted_nodes = json.loads(content)
            if not isinstance(extracted_nodes, list):
                logger.warning(f"DeepSleep: LLM returned non-list JSON for {node.path.stem}")
                return False

            if not extracted_nodes:
                logger.debug(f"DeepSleep: No concepts extracted from {node.path.stem}")
                return True  # Empty but valid — still a success

            for en in extracted_nodes:
                raw_name = en.get("name", "")
                name = sanitize_node_name(raw_name)
                if not name:
                    continue
                
                new_content = en.get("content", "")
                connections = en.get("connections", [])
                
                try:
                    existing = self.vault.read_node(name)
                    # Merge content
                    updated_content = existing.content + "\n\n" + new_content
                    # Merge connections metadata
                    updated_meta = dict(existing.metadata)
                    existing_connections = updated_meta.get("connections", [])
                    merged_connections = list(set(existing_connections + connections))
                    updated_meta["connections"] = merged_connections
                    self.vault.write_node(name, updated_content, updated_meta)
                    self.index.index_node(name, updated_content, node_type="concept")
                except FileNotFoundError:
                    metadata = {"type": "concept", "connections": connections}
                    self.vault.write_node(name, new_content, metadata)
                    self.index.index_node(name, new_content, node_type="concept")

            return True

        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.error(f"DeepSleep: Failed to parse LLM output for {node.path.stem}: {e}")
            return False

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
        self.index.index_node("index", index_content, node_type="index")
