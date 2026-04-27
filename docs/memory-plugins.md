# Custom Memory Plugins

By default, nanobot uses a file-based memory system storing data in `history.jsonl` and editing `MEMORY.md`, `SOUL.md`, and `USER.md` (see [Memory](./memory.md) for details).

However, you can completely replace this system with a custom **Memory Plugin**. This allows you to integrate nanobot with external databases (e.g., PostgreSQL, MongoDB, Redis), vector databases, cloud storage APIs, or custom agent-framework memory mechanisms.

## Configuring a Plugin

To load a custom memory plugin, you must configure it in your `config.json` (or active environment configuration).

```json
{
  "agents": {
    "defaults": {
      "memory_plugin": "my_package.my_module.MyCustomPlugin"
    }
  }
}
```

The string should be a standard Python dotted import path pointing to a class that implements the `MemoryPlugin` interface. Nanobot will automatically resolve and instantiate this class at startup. If the configuration is omitted or null, nanobot defaults to its standard `FileMemoryPlugin`.

## The `MemoryPlugin` Interface

Custom plugins must implement the `MemoryPlugin` abstract base class defined in `nanobot.memory.base`.

The interface encompasses three primary responsibilities:
1. **Durable Knowledge Management**: Reading and writing the agent's long-term documents (`MEMORY.md`, `SOUL.md`, `USER.md`).
2. **Archival History Management**: Handling the append-only stream of historical turns.
3. **The Dream Operation**: Analyzing archival history to update durable knowledge.

### Durable Knowledge
These methods control the long-term context that is injected into the agent's prompt on every turn.

- `read_memory() -> str`: Returns the contents of the project memory (`MEMORY.md`).
- `write_memory(content: str) -> None`: Updates the project memory.
- `read_soul() -> str`: Returns the agent's system prompt instructions (`SOUL.md`).
- `read_user_profile() -> str`: Returns the user's profile (`USER.md`).

### Raw History Archive
These methods manage the summarized, append-only history log.

- `raw_archive(messages: list[dict], *, max_chars: int | None = None) -> None`: Archives a batch of raw messages.
- `archive_summary(summary: str, *, max_chars: int | None = None) -> None`: Archives a text summary block.
- `append_history(entry: str, *, max_chars: int | None = None) -> int`: Appends a single raw entry to the history archive and returns its cursor/index.
- `read_history(max_entries: int) -> list[dict]`: Reads the most recent N archived messages to inject into the active context.

### Dream Support Primitives
These methods are used to track and fetch the difference between what history has been generated and what history has been processed by Dream.

- `get_last_dream_cursor() -> int`: Gets the cursor position of the last successful Dream run.
- `set_last_dream_cursor(value: int) -> None`: Saves the cursor position of the latest Dream run.
- `read_unprocessed_history(since_cursor: int) -> list[dict]`: Gets history messages starting after the given cursor.
- `annotate_memory_with_ages() -> str`: Reads the memory document, optionally annotating it with age or source attribution (like `git blame`).

### The Dream Operation
These methods execute the background consolidation process.

- `set_provider(provider, model: str) -> None`: Injects the active LLM provider into the memory plugin. This allows the plugin to use the LLM to perform its Dream operations.
- `async dream() -> bool`: Runs memory consolidation. Returns `True` if work was done, `False` if nothing needed processing.

### Versioning (Optional)
If your backend supports versioning (e.g. Git or snapshotting), you can implement these methods:

- `is_versioned() -> bool`: Return `True` if this plugin supports history snapshots.
- `commit_memory_snapshot(message: str) -> str | None`: Commits the current memory state and returns a unique snapshot identifier (e.g., a SHA).

## Example: A Null Plugin

If you wanted to build an agent that *never* remembers anything long-term, you could implement a "Null" Memory Plugin that stores nothing.

```python
from nanobot.memory.base import MemoryPlugin

class NullMemoryPlugin(MemoryPlugin):
    def read_memory(self) -> str: return ""
    def write_memory(self, content: str) -> None: pass
    def read_soul(self) -> str: return "I am an amnesiac bot."
    def read_user_profile(self) -> str: return ""
    
    def raw_archive(self, messages, *, max_chars=None) -> None: pass
    def archive_summary(self, summary, *, max_chars=None) -> None: pass
    def append_history(self, entry, *, max_chars=None) -> int: return 0
    def read_history(self, max_entries: int) -> list[dict]: return []
    
    def get_last_dream_cursor(self) -> int: return 0
    def set_last_dream_cursor(self, value: int) -> None: pass
    def read_unprocessed_history(self, since_cursor: int) -> list[dict]: return []
    def annotate_memory_with_ages(self) -> str: return ""
    def commit_memory_snapshot(self, message: str) -> str | None: return None
    
    async def dream(self) -> bool: return False
```

Once saved somewhere in your python path (e.g. `plugins/null_memory.py`), you configure nanobot to use it via `"memory_plugin": "plugins.null_memory.NullMemoryPlugin"`.
