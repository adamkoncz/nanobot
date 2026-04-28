import re
import yaml
from pathlib import Path
from typing import Any, NamedTuple


class VaultNode(NamedTuple):
    path: Path
    content: str
    metadata: dict[str, Any]
    links: list[str]


# Characters not allowed in node names — prevents path traversal and filesystem issues.
_UNSAFE_NAME_RE = re.compile(r"[^a-z0-9_\-]")


def sanitize_node_name(name: str) -> str:
    """Normalise an arbitrary string into a safe, flat filename stem.

    - Lowercases
    - Replaces spaces with underscores
    - Strips anything that isn't alphanumeric, underscore, or hyphen
    - Collapses repeated underscores
    - Returns empty string if nothing survives (caller must check)
    """
    slug = name.strip().lower().replace(" ", "_")
    slug = _UNSAFE_NAME_RE.sub("", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug


class Vault:
    """
    Manages atomic Markdown files with YAML frontmatter and [[Wikilinks]].
    """

    WIKILINK_RE = re.compile(r"\[\[(.*?)\]\]")
    FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)

    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, name: str) -> Path:
        """Resolve a node name to a file path, enforcing it stays within the vault."""
        path = (self.directory / f"{name}.md").resolve()
        if not str(path).startswith(str(self.directory.resolve())):
            raise ValueError(f"Node name escapes vault directory: {name!r}")
        return path

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
        path = self._resolve_path(name)
        return self._parse_file(path)

    def write_node(self, name: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        """Writes a node with optional frontmatter."""
        path = self._resolve_path(name)
        
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
        path = self._resolve_path(name)
        if path.exists():
            path.unlink()
