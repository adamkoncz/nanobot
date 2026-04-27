from abc import ABC, abstractmethod


class MemoryPlugin(ABC):
    """Universal interface for Nanobot memory storage and dreaming."""

    # -- Long-term memory (MEMORY.md, SOUL.md, USER.md) --
    
    @abstractmethod
    def read_memory(self) -> str:
        """Read the long-term memory document."""
        ...

    @abstractmethod
    def write_memory(self, content: str) -> None:
        """Write the long-term memory document."""
        ...

    @abstractmethod
    def read_soul(self) -> str:
        """Read the system identity/soul document."""
        ...

    @abstractmethod
    def read_user_profile(self) -> str:
        """Read the user profile document."""
        ...

    # -- Raw history archive (history.jsonl) --
    
    @abstractmethod
    def raw_archive(self, messages: list[dict], *, max_chars: int | None = None) -> None:
        """Archive a batch of raw session messages to long-term storage."""
        ...

    @abstractmethod
    def archive_summary(self, summary: str, *, max_chars: int | None = None) -> None:
        """Archive a summarized block of messages."""
        ...

    @abstractmethod
    def append_history(self, entry: str, *, max_chars: int | None = None) -> int:
        """Append a single entry to the history archive and return its auto-incrementing cursor."""
        ...

    @abstractmethod
    def read_history(self, max_entries: int) -> list[dict]:
        """Read the most recent N archived messages from long-term storage."""
        ...

    @abstractmethod
    def get_last_dream_cursor(self) -> int:
        """Get the cursor position (e.g. index/offset) of the last Dream run."""
        ...

    @abstractmethod
    def set_last_dream_cursor(self, value: int) -> None:
        """Save the cursor position of the latest Dream run."""
        ...

    # -- Git / versioning (optional; default = no-op) --
    
    def is_versioned(self) -> bool:
        """Whether this plugin supports version control/history (e.g. git)."""
        return False

    # -- Dream support primitives --
    
    @abstractmethod
    def read_unprocessed_history(self, since_cursor: int) -> list[dict]:
        """Get unprocessed history messages starting after the given cursor."""
        ...

    @abstractmethod
    def annotate_memory_with_ages(self) -> str:
        """Read the memory document, optionally annotating it with ages (e.g. via git blame)."""
        ...

    @abstractmethod
    def commit_memory_snapshot(self, message: str) -> str | None:
        """Commit the current memory state to version control (if supported) and return the SHA."""
        ...

    # -- The Dream Operation --
    
    def set_provider(self, provider: 'nanobot.providers.base.LLMProvider', model: str) -> None:
        """Inject LLM provider into the memory plugin for tasks like dreaming."""
        pass

    @abstractmethod
    async def dream(self) -> bool:
        """
        Run memory consolidation (Dream).
        Returns True if work was done, False if nothing needed processing.
        """
        ...
