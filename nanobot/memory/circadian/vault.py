import re
import yaml
from pathlib import Path
from typing import Any, NamedTuple


class VaultNode(NamedTuple):
    path: Path
    content: str
    metadata: dict[str, Any]
    links: list[str]


class Vault:
    """
    Manages atomic Markdown files with YAML frontmatter and [[Wikilinks]].
    """

    WIKILINK_RE = re.compile(r"\[\[(.*?)\]\]")
    FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)

    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def _parse_file(self, path: Path) -> VaultNode:
        if not path.exists():
            raise FileNotFoundError(f"Vault node not found: {path}")

        raw_content = path.read_text(encoding="utf-8")
        
        # Extract frontmatter
        metadata = {}
        content = raw_content
        fm_match = self.FRONTMATTER_RE.match(raw_content)
        if fm_match:
            try:
                metadata = yaml.safe_load(fm_match.group(1)) or {}
            except yaml.YAMLError:
                pass
            content = raw_content[fm_match.end():]
            
        # Extract wikilinks
        links = self.WIKILINK_RE.findall(content)
        
        return VaultNode(
            path=path,
            content=content.strip(),
            metadata=metadata,
            links=links
        )

    def read_node(self, name: str) -> VaultNode:
        """Reads a node by its name (without extension)."""
        path = self.directory / f"{name}.md"
        return self._parse_file(path)

    def write_node(self, name: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        """Writes a node with optional frontmatter."""
        path = self.directory / f"{name}.md"
        
        output = ""
        if metadata:
            output += "---\n"
            output += yaml.safe_dump(metadata, default_flow_style=False, sort_keys=False)
            output += "---\n\n"
            
        output += content
        path.write_text(output, encoding="utf-8")

    def list_nodes(self) -> list[str]:
        """Returns a list of node names."""
        return [p.stem for p in self.directory.glob("*.md")]
        
    def delete_node(self, name: str) -> None:
        path = self.directory / f"{name}.md"
        if path.exists():
            path.unlink()
