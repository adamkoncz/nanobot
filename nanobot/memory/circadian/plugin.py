from pathlib import Path
from typing import Any, TYPE_CHECKING

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

    # -- Long-term memory --

    def read_memory(self) -> str:
        """
        In Circadian mode, memory is distributed. For prompt injection,
        we might compile a digest or return the root index.
        """
        index_file = self.vault_dir / "index.md"
        if index_file.exists():
            return index_file.read_text(encoding="utf-8")
        return "Circadian Vault initialized."

    def write_memory(self, content: str) -> None:
        """Fallback for direct memory writes."""
        index_file = self.vault_dir / "index.md"
        index_file.write_text(content, encoding="utf-8")

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
        # Save config for our sleep phases (to be implemented)
        pass

    def set_provider(self, provider: "LLMProvider", model: str) -> None:
        self._provider = provider
        self._model = model

    async def dream(self) -> bool:
        """
        Run the Circadian sleep cycle (Light, Deep, REM).
        """
        from nanobot.memory.circadian.phases import LightSleepPhase, DeepSleepPhase, REMSleepPhase
        from loguru import logger
        
        logger.info("Starting Circadian Sleep Cycle...")
        
        # 1. Light Sleep
        light = LightSleepPhase(self)
        processed = await light.run()
        
        # 2. Deep Sleep
        deep = DeepSleepPhase(self)
        promoted = await deep.run()
        
        # 3. REM Sleep
        # In a real setup, we would respect config.dreaming.rem_enabled
        # and token limits here. For now we run it directly.
        rem = REMSleepPhase(self)
        await rem.run()
        
        logger.info("Circadian Sleep Cycle completed.")
        return processed > 0 or promoted > 0
