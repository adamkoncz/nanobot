from pathlib import Path
from typing import Any

from nanobot.agent.memory import Dream, MemoryStore
from nanobot.memory.base import MemoryPlugin


class FileMemoryPlugin(MemoryPlugin):
    """Default file-based memory implementation wrapping MemoryStore and Dream."""

    def __init__(self, workspace: Path, **kwargs: Any):
        self._store = MemoryStore(workspace, **kwargs)
        self._dream = Dream(store=self._store, provider=None, model="") # Provider will be injected

    # -- Long-term memory --

    def read_memory(self) -> str:
        return self._store.read_memory()

    def write_memory(self, content: str) -> None:
        self._store.write_memory(content)

    def read_soul(self) -> str:
        return self._store.read_soul()

    def read_user_profile(self) -> str:
        return self._store.read_user()

    # -- Raw history archive --

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
        content = self._store.read_memory()
        if self._dream.annotate_line_ages:
            return self._dream._annotate_with_ages(content)
        return content

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
            self._dream.model = model_override
        if max_batch_size is not None:
            self._dream.max_batch_size = max_batch_size
        if max_iterations is not None:
            self._dream.max_iterations = max_iterations
        if annotate_line_ages is not None:
            self._dream.annotate_line_ages = annotate_line_ages

    def set_provider(self, provider: 'nanobot.providers.base.LLMProvider', model: str) -> None:
        self._dream.set_provider(provider, model)

    async def dream(self) -> bool:
        return await self._dream.run()
