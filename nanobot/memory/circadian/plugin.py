from pathlib import Path
from typing import Any, TYPE_CHECKING

from loguru import logger

from nanobot.agent.memory import MemoryStore
from nanobot.memory.base import MemoryPlugin

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider


class CircadianMemoryPlugin(MemoryPlugin):
    """
    Circadian memory plugin: an atomic, Obsidian-style Markdown vault
    maintained by background biological-inspired sleep cycles.
    """

    def __init__(self, workspace: Path, **kwargs: Any):
        self._store = MemoryStore(workspace, **kwargs)
        self.workspace = workspace
        self.vault_dir = workspace / "vault"
        self.vault_dir.mkdir(parents=True, exist_ok=True)
        
        from nanobot.memory.circadian.vault import Vault
        from nanobot.memory.circadian.index import CircadianIndex
        
        self.vault = Vault(self.vault_dir)
        self.index = CircadianIndex(self.vault_dir / "index.db")
        
        self._provider: "LLMProvider | None" = None
        self._model = ""

        # Dream configuration — populated by configure_dream() / gateway
        self._rem_enabled: bool = False
        self._rem_token_limit: int = 10_000
        self._model_override: str | None = None

    # -- Long-term memory --

    def read_memory(self) -> str:
        """
        In Circadian mode, memory is distributed. For prompt injection,
        we return the root index node's content (without frontmatter).
        """
        try:
            node = self.vault.read_node("index")
            return node.content
        except FileNotFoundError:
            return "Circadian Vault initialized."

    def write_memory(self, content: str) -> None:
        """Fallback for direct memory writes — routes through the Vault."""
        try:
            existing = self.vault.read_node("index")
            metadata = existing.metadata
        except FileNotFoundError:
            metadata = {"type": "index"}
        self.vault.write_node("index", content, metadata)
        self.index.index_node("index", content, node_type="index")

    def read_soul(self) -> str:
        return self._store.read_soul()

    def read_user_profile(self) -> str:
        return self._store.read_user()

    # -- Raw history archive --
    # Delegate to MemoryStore which handles history.jsonl perfectly

    def raw_archive(self, messages: list[dict], *, max_chars: int | None = None) -> None:
        self._store.raw_archive(messages, max_chars=max_chars)

    def archive_summary(self, summary: str, *, max_chars: int | None = None) -> None:
        self._store.append_history(summary, max_chars=max_chars)

    def append_history(self, entry: str, *, max_chars: int | None = None) -> int:
        return self._store.append_history(entry, max_chars=max_chars)

    def read_history(self, max_entries: int) -> list[dict]:
        entries = self._store._read_entries()
        return entries[-max_entries:] if max_entries > 0 else entries

    def get_last_dream_cursor(self) -> int:
        return self._store.get_last_dream_cursor()

    def set_last_dream_cursor(self, value: int) -> None:
        self._store.set_last_dream_cursor(value)

    # -- Git / versioning --

    def is_versioned(self) -> bool:
        return True

    def commit_memory_snapshot(self, message: str) -> str | None:
        return self._store.git.commit(message)

    # -- Dream support primitives --

    def read_unprocessed_history(self, since_cursor: int) -> list[dict]:
        return self._store.read_unprocessed_history(since_cursor)

    def annotate_memory_with_ages(self) -> str:
        return self.read_memory()

    # -- The Dream Operation --

    def configure_dream(
        self,
        *,
        model_override: str | None = None,
        max_batch_size: int | None = None,
        max_iterations: int | None = None,
        annotate_line_ages: bool | None = None,
    ) -> None:
        if model_override is not None:
            self._model_override = model_override
            logger.debug(f"Circadian dream: model override set to {model_override}")
        # max_batch_size, max_iterations, annotate_line_ages are not applicable
        # to the Circadian model but we accept them silently for interface compat.

    def set_provider(self, provider: "LLMProvider", model: str) -> None:
        self._provider = provider
        self._model = model

    def set_rem_config(self, *, enabled: bool, token_limit: int) -> None:
        """Configure REM phase from DreamConfig."""
        self._rem_enabled = enabled
        self._rem_token_limit = token_limit

    async def dream(self) -> bool:
        """
        Run the Circadian sleep cycle (Light, Deep, REM).
        """
        from nanobot.memory.circadian.phases.light import LightSleepPhase
        from nanobot.memory.circadian.phases.deep import DeepSleepPhase
        from nanobot.memory.circadian.phases.rem import REMSleepPhase
        
        model = self._model_override or self._model
        
        logger.info("Starting Circadian Sleep Cycle...")
        
        # 1. Light Sleep — ingest history into staging nodes
        light = LightSleepPhase(self)
        processed = await light.run()
        
        # 2. Deep Sleep — promote staging into atomic concepts
        deep = DeepSleepPhase(self)
        promoted = await deep.run()
        
        # 3. REM Sleep — creative synthesis (respects config)
        if self._rem_enabled:
            rem = REMSleepPhase(self, token_limit=self._rem_token_limit)
            await rem.run()
        else:
            logger.debug("Circadian REM phase is disabled.")
        
        logger.info("Circadian Sleep Cycle completed.")
        return processed > 0 or promoted > 0

    def close(self) -> None:
        """Release resources (SQLite connection)."""
        self.index.close()
