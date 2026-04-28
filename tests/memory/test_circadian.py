import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from nanobot.memory.circadian.vault import Vault, sanitize_node_name
from nanobot.memory.circadian.index import CircadianIndex
from nanobot.memory.circadian.plugin import CircadianMemoryPlugin


# ── Vault tests ─────────────────────────────────────────────────────────────

class TestVault:
    def test_basic_operations(self, tmp_path: Path):
        vault = Vault(tmp_path / "vault")

        metadata = {"type": "test", "tags": ["a", "b"]}
        content = "This is a test with a [[Wikilink]]."
        vault.write_node("test_node", content, metadata)

        node = vault.read_node("test_node")
        assert node.content == content
        assert node.metadata == metadata
        assert "Wikilink" in node.links

        assert "test_node" in vault.list_nodes()

        vault.delete_node("test_node")
        assert not vault.list_nodes()

    def test_read_nonexistent_raises(self, tmp_path: Path):
        vault = Vault(tmp_path / "vault")
        with pytest.raises(FileNotFoundError):
            vault.read_node("does_not_exist")

    def test_path_traversal_blocked(self, tmp_path: Path):
        vault = Vault(tmp_path / "vault")
        with pytest.raises(ValueError, match="escapes vault"):
            vault.read_node("../../etc/passwd")

    def test_write_without_metadata(self, tmp_path: Path):
        vault = Vault(tmp_path / "vault")
        vault.write_node("bare", "Just content, no frontmatter.")
        node = vault.read_node("bare")
        assert node.content == "Just content, no frontmatter."
        assert node.metadata == {}

    def test_wikilinks_extraction(self, tmp_path: Path):
        vault = Vault(tmp_path / "vault")
        vault.write_node("multi", "See [[Alpha]] and [[Beta]].")
        node = vault.read_node("multi")
        assert node.links == ["Alpha", "Beta"]


class TestSanitizeNodeName:
    def test_basic(self):
        assert sanitize_node_name("Hello World") == "hello_world"

    def test_strips_traversal(self):
        assert sanitize_node_name("../../etc/passwd") == "etcpasswd"

    def test_strips_special_chars(self):
        assert sanitize_node_name("C++ Templates!") == "c_templates"

    def test_empty_input(self):
        assert sanitize_node_name("") == ""
        assert sanitize_node_name("!!!") == ""


# ── Index tests ─────────────────────────────────────────────────────────────

class TestCircadianIndex:
    def test_basic_index_and_search(self, tmp_path: Path):
        index = CircadianIndex(tmp_path / "index.db")

        index.index_node("node1", "Hello world from index")
        index.index_node("node2", "Another test node")

        results = index.search("Hello")
        assert len(results) == 1
        assert results[0]["name"] == "node1"

        index.remove_node("node1")
        results = index.search("Hello")
        assert len(results) == 0
        index.close()

    def test_fts_special_characters_safe(self, tmp_path: Path):
        index = CircadianIndex(tmp_path / "index.db")
        index.index_node("cpp", "C++ templates and pointers")

        # These should NOT crash
        results = index.search("C++")
        assert isinstance(results, list)
        results = index.search('NOT "something"')
        assert isinstance(results, list)
        results = index.search("(parentheses)")
        assert isinstance(results, list)
        index.close()

    def test_list_by_type(self, tmp_path: Path):
        index = CircadianIndex(tmp_path / "index.db")
        index.index_node("inbox_1", "staging content", node_type="staging")
        index.index_node("python", "a language", node_type="concept")
        index.index_node("inbox_2", "more staging", node_type="staging")

        staging = index.list_by_type("staging")
        assert set(staging) == {"inbox_1", "inbox_2"}

        concepts = index.list_by_type("concept")
        assert concepts == ["python"]
        index.close()

    def test_upsert_updates_content(self, tmp_path: Path):
        index = CircadianIndex(tmp_path / "index.db")
        index.index_node("topic", "Version 1")
        index.index_node("topic", "Version 2")

        results = index.search("Version")
        assert len(results) == 1
        assert "Version 2" in results[0]["content"]
        index.close()


# ── Plugin tests ────────────────────────────────────────────────────────────

class TestPlugin:
    def test_initialization(self, tmp_path: Path):
        plugin = CircadianMemoryPlugin(tmp_path)
        assert plugin.vault_dir.exists()
        assert plugin.vault is not None
        assert plugin.index is not None
        plugin.close()

    def test_history_delegation(self, tmp_path: Path):
        plugin = CircadianMemoryPlugin(tmp_path)
        plugin.append_history("test history entry")
        entries = plugin.read_history(10)
        assert len(entries) == 1
        assert entries[0]["content"] == "test history entry"
        plugin.close()

    def test_read_memory_without_frontmatter(self, tmp_path: Path):
        """read_memory() must return clean content, not raw YAML frontmatter."""
        plugin = CircadianMemoryPlugin(tmp_path)
        plugin.vault.write_node("index", "# My Index", {"type": "index"})

        memory = plugin.read_memory()
        assert "---" not in memory
        assert "type: index" not in memory
        assert "# My Index" in memory
        plugin.close()

    def test_write_memory_preserves_metadata(self, tmp_path: Path):
        plugin = CircadianMemoryPlugin(tmp_path)
        plugin.vault.write_node("index", "old content", {"type": "index"})
        plugin.write_memory("new content")

        node = plugin.vault.read_node("index")
        assert node.content == "new content"
        assert node.metadata.get("type") == "index"
        plugin.close()

    def test_rem_disabled_by_default(self, tmp_path: Path):
        plugin = CircadianMemoryPlugin(tmp_path)
        assert plugin._rem_enabled is False
        plugin.close()

    def test_set_rem_config(self, tmp_path: Path):
        plugin = CircadianMemoryPlugin(tmp_path)
        plugin.set_rem_config(enabled=True, token_limit=5000)
        assert plugin._rem_enabled is True
        assert plugin._rem_token_limit == 5000
        plugin.close()

    def test_configure_dream_stores_model(self, tmp_path: Path):
        plugin = CircadianMemoryPlugin(tmp_path)
        plugin.configure_dream(model_override="gpt-4o")
        assert plugin._model_override == "gpt-4o"
        plugin.close()


# ── Async phase tests ───────────────────────────────────────────────────────

def _make_mock_provider(response_content: str, usage: dict | None = None):
    """Create a mock LLMProvider with chat_with_retry."""
    provider = MagicMock()
    mock_response = MagicMock()
    mock_response.content = response_content
    mock_response.finish_reason = "stop"
    mock_response.usage = usage or {}
    provider.chat_with_retry = AsyncMock(return_value=mock_response)
    provider.chat = AsyncMock(return_value=mock_response)
    return provider


class TestLightSleepPhase:
    @pytest.mark.asyncio
    async def test_stages_history(self, tmp_path: Path):
        from nanobot.memory.circadian.phases.light import LightSleepPhase

        plugin = CircadianMemoryPlugin(tmp_path)
        plugin._provider = _make_mock_provider("## Summary\nUser asked about Python.")
        plugin._model = "test-model"

        # Seed some history with cursor values
        plugin.append_history("user asked about Python")

        light = LightSleepPhase(plugin)
        count = await light.run()

        assert count > 0
        staging = plugin.index.list_by_type("staging")
        assert len(staging) == 1
        assert staging[0].startswith("inbox_")
        plugin.close()

    @pytest.mark.asyncio
    async def test_no_history_noop(self, tmp_path: Path):
        from nanobot.memory.circadian.phases.light import LightSleepPhase

        plugin = CircadianMemoryPlugin(tmp_path)
        light = LightSleepPhase(plugin)
        count = await light.run()
        assert count == 0
        plugin.close()

    @pytest.mark.asyncio
    async def test_llm_error_falls_back_to_raw(self, tmp_path: Path):
        from nanobot.memory.circadian.phases.light import LightSleepPhase

        plugin = CircadianMemoryPlugin(tmp_path)
        error_provider = MagicMock()
        error_response = MagicMock()
        error_response.content = "Error: rate limit"
        error_response.finish_reason = "error"
        error_response.usage = {}
        error_provider.chat_with_retry = AsyncMock(return_value=error_response)
        
        plugin._provider = error_provider
        plugin._model = "test-model"
        plugin.append_history("some entry")

        light = LightSleepPhase(plugin)
        count = await light.run()
        assert count > 0

        # Should have staged raw text, not the error message
        staging = plugin.index.list_by_type("staging")
        assert len(staging) == 1
        plugin.close()


class TestDeepSleepPhase:
    @pytest.mark.asyncio
    async def test_promotes_staging_to_concepts(self, tmp_path: Path):
        from nanobot.memory.circadian.phases.deep import DeepSleepPhase

        plugin = CircadianMemoryPlugin(tmp_path)
        plugin.vault.write_node("inbox_test", "User discussed Python and SQLite.", {
            "type": "staging", "status": "unprocessed"
        })
        plugin.index.index_node("inbox_test", "User discussed Python and SQLite.", node_type="staging")

        llm_output = json.dumps([
            {"name": "Python", "content": "A programming language.", "connections": ["SQLite"]},
            {"name": "SQLite", "content": "An embedded database.", "connections": ["Python"]},
        ])
        plugin._provider = _make_mock_provider(llm_output)
        plugin._model = "test-model"

        deep = DeepSleepPhase(plugin)
        count = await deep.run()

        assert count == 1  # 1 staging node processed
        assert "python" in plugin.vault.list_nodes()
        assert "sqlite" in plugin.vault.list_nodes()
        assert "inbox_test" not in plugin.vault.list_nodes()
        plugin.close()

    @pytest.mark.asyncio
    async def test_failed_parse_preserves_staging(self, tmp_path: Path):
        from nanobot.memory.circadian.phases.deep import DeepSleepPhase

        plugin = CircadianMemoryPlugin(tmp_path)
        plugin.vault.write_node("inbox_bad", "Content", {
            "type": "staging", "status": "unprocessed"
        })
        plugin.index.index_node("inbox_bad", "Content", node_type="staging")

        # Return garbage that won't parse as JSON
        plugin._provider = _make_mock_provider("This is not JSON at all!")
        plugin._model = "test-model"

        deep = DeepSleepPhase(plugin)
        count = await deep.run()

        assert count == 0  # Not counted as success
        assert "inbox_bad" in plugin.vault.list_nodes()  # Preserved!

        node = plugin.vault.read_node("inbox_bad")
        assert node.metadata.get("status") == "failed"
        plugin.close()

    @pytest.mark.asyncio
    async def test_merges_connections_on_update(self, tmp_path: Path):
        from nanobot.memory.circadian.phases.deep import DeepSleepPhase

        plugin = CircadianMemoryPlugin(tmp_path)

        # Pre-existing concept with connections
        plugin.vault.write_node("python", "A language.", {
            "type": "concept", "connections": ["SQLite"]
        })
        plugin.index.index_node("python", "A language.", node_type="concept")

        # Staging node that will update python with new connection
        plugin.vault.write_node("inbox_merge", "More Python info", {
            "type": "staging", "status": "unprocessed"
        })
        plugin.index.index_node("inbox_merge", "More Python info", node_type="staging")

        llm_output = json.dumps([
            {"name": "Python", "content": "Also used for AI.", "connections": ["TensorFlow"]},
        ])
        plugin._provider = _make_mock_provider(llm_output)
        plugin._model = "test-model"

        deep = DeepSleepPhase(plugin)
        await deep.run()

        node = plugin.vault.read_node("python")
        assert "SQLite" in node.metadata["connections"]
        assert "TensorFlow" in node.metadata["connections"]
        plugin.close()


class TestREMSleepPhase:
    @pytest.mark.asyncio
    async def test_insight_prefixed(self, tmp_path: Path):
        from nanobot.memory.circadian.phases.rem import REMSleepPhase

        plugin = CircadianMemoryPlugin(tmp_path)
        plugin.vault.write_node("python", "A language.", {"type": "concept"})
        plugin.vault.write_node("sqlite", "A database.", {"type": "concept"})
        plugin.index.index_node("python", "A language.", node_type="concept")
        plugin.index.index_node("sqlite", "A database.", node_type="concept")

        plugin._provider = _make_mock_provider(
            "# Data and Code\n\nPython and SQLite connect through data management."
        )
        plugin._model = "test-model"

        rem = REMSleepPhase(plugin)
        count = await rem.run()

        assert count == 1
        insight_nodes = [n for n in plugin.vault.list_nodes() if n.startswith("insight_")]
        assert len(insight_nodes) == 1
        # Must NOT overwrite the concept nodes
        assert "python" in plugin.vault.list_nodes()
        assert "sqlite" in plugin.vault.list_nodes()
        plugin.close()

    @pytest.mark.asyncio
    async def test_too_few_concepts_noop(self, tmp_path: Path):
        from nanobot.memory.circadian.phases.rem import REMSleepPhase

        plugin = CircadianMemoryPlugin(tmp_path)
        plugin.vault.write_node("only_one", "Solo node.", {"type": "concept"})
        plugin._provider = _make_mock_provider("unused")
        plugin._model = "test-model"

        rem = REMSleepPhase(plugin)
        count = await rem.run()
        assert count == 0
        plugin.close()
