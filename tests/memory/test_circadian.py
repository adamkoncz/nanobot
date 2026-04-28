import pytest
from pathlib import Path
from nanobot.memory.circadian.vault import Vault
from nanobot.memory.circadian.index import CircadianIndex
from nanobot.memory.circadian.plugin import CircadianMemoryPlugin


def test_vault_operations(tmp_path: Path):
    vault = Vault(tmp_path / "vault")
    
    # Write
    metadata = {"type": "test", "tags": ["a", "b"]}
    content = "This is a test with a [[Wikilink]]."
    vault.write_node("test_node", content, metadata)
    
    # Read
    node = vault.read_node("test_node")
    assert node.content == content
    assert node.metadata == metadata
    assert "Wikilink" in node.links
    
    # List
    nodes = vault.list_nodes()
    assert "test_node" in nodes
    
    # Delete
    vault.delete_node("test_node")
    assert not vault.list_nodes()


def test_circadian_index(tmp_path: Path):
    index = CircadianIndex(tmp_path / "index.db")
    
    # Indexing
    index.index_node("node1", "Hello world from index")
    index.index_node("node2", "Another test node")
    
    # Keyword search
    results = index.search("Hello")
    assert len(results) == 1
    assert results[0]["name"] == "node1"
    
    # Remove
    index.remove_node("node1")
    results = index.search("Hello")
    assert len(results) == 0


def test_circadian_plugin_initialization(tmp_path: Path):
    plugin = CircadianMemoryPlugin(tmp_path)
    assert plugin.vault_dir.exists()
    assert plugin.vault is not None
    assert plugin.index is not None

    # Test raw archiving delegates to MemoryStore correctly
    plugin.append_history("test history entry")
    entries = plugin.read_history(10)
    assert len(entries) == 1
    assert entries[0]["content"] == "test history entry"
