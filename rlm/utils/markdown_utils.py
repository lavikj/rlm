"""
Markdown utilities for intelligent document chunking.

These utilities are exposed in the REPL environment for LLMs to use
when processing markdown documents.
"""

import re
from dataclasses import dataclass


@dataclass
class MarkdownChunk:
    """A chunk of markdown content with its header hierarchy."""

    header: str  # The header text (e.g., "Introduction")
    level: int  # Header level (1-6 for h1-h6, 0 for content before any header)
    content: str  # The content under this header (not including sub-headers)
    full_content: str  # Content including all nested sub-sections
    path: list[str]  # Header hierarchy path (e.g., ["Chapter 1", "Section 1.1"])
    start_line: int  # Line number where this chunk starts

    def __str__(self) -> str:
        path_str = " > ".join(self.path) if self.path else "(root)"
        return f"[{path_str}] ({len(self.content)} chars)"

    def to_dict(self) -> dict:
        return {
            "header": self.header,
            "level": self.level,
            "content": self.content,
            "full_content": self.full_content,
            "path": self.path,
            "start_line": self.start_line,
        }


def chunk_markdown(
    text: str,
    min_level: int = 1,
    max_level: int = 6,
    include_content_before_headers: bool = True,
) -> list[MarkdownChunk]:
    """
    Intelligently chunk markdown text by headers.

    This function parses markdown and splits it into chunks based on header
    structure (# through ######). Each chunk contains the header, its content,
    and the full hierarchy path.

    Args:
        text: The markdown text to chunk.
        min_level: Minimum header level to split on (1 = h1, 2 = h2, etc.).
                   Default 1 includes all headers.
        max_level: Maximum header level to split on. Default 6 includes all.
        include_content_before_headers: If True, include any content that
                   appears before the first header as a chunk with level 0.

    Returns:
        List of MarkdownChunk objects, each containing:
        - header: The header text
        - level: Header level (1-6, or 0 for pre-header content)
        - content: Direct content under this header (excluding sub-sections)
        - full_content: All content including nested sub-sections
        - path: List of headers forming the hierarchy path
        - start_line: Line number where chunk starts

    Example:
        >>> text = '''
        ... # Chapter 1
        ... Introduction text.
        ...
        ... ## Section 1.1
        ... Section content.
        ...
        ... # Chapter 2
        ... More content.
        ... '''
        >>> chunks = chunk_markdown(text)
        >>> for chunk in chunks:
        ...     print(f"{chunk.path}: {len(chunk.content)} chars")
        ['Chapter 1']: 19 chars
        ['Chapter 1', 'Section 1.1']: 17 chars
        ['Chapter 2']: 14 chars
    """
    lines = text.split("\n")
    chunks: list[MarkdownChunk] = []

    # Regex to match markdown headers (# to ######)
    header_pattern = re.compile(r"^(#{1,6})\s+(.+)$")

    # Track current position and header stack
    current_content_lines: list[str] = []
    current_header: str | None = None
    current_level: int = 0
    current_start_line: int = 0
    header_stack: list[tuple[str, int]] = []  # (header_text, level)

    def flush_chunk():
        """Save the current accumulated content as a chunk."""
        nonlocal current_content_lines, current_header, current_level, current_start_line

        content = "\n".join(current_content_lines).strip()

        # Build path from header stack
        path = [h for h, _ in header_stack]
        if current_header and (not path or path[-1] != current_header):
            path.append(current_header)

        if content or current_header:
            chunks.append(
                MarkdownChunk(
                    header=current_header or "",
                    level=current_level,
                    content=content,
                    full_content=content,  # Will be updated later
                    path=path,
                    start_line=current_start_line,
                )
            )

        current_content_lines = []

    for line_num, line in enumerate(lines):
        match = header_pattern.match(line)

        if match:
            header_markers = match.group(1)
            header_text = match.group(2).strip()
            level = len(header_markers)

            # Check if this header level is in our range
            if min_level <= level <= max_level:
                # Flush previous content
                flush_chunk()

                # Update header stack - pop headers at same or deeper level
                while header_stack and header_stack[-1][1] >= level:
                    header_stack.pop()

                # Push current header onto stack
                header_stack.append((header_text, level))

                current_header = header_text
                current_level = level
                current_start_line = line_num
            else:
                # Header outside our range - treat as content
                current_content_lines.append(line)
        else:
            current_content_lines.append(line)

    # Flush final chunk
    flush_chunk()

    # Filter out empty pre-header content if not wanted
    if not include_content_before_headers and chunks and chunks[0].level == 0:
        if not chunks[0].content.strip():
            chunks.pop(0)

    # Calculate full_content for each chunk (content + all nested sub-sections)
    _calculate_full_content(chunks)

    return chunks


def _calculate_full_content(chunks: list[MarkdownChunk]) -> None:
    """Calculate full_content for each chunk including nested sub-sections."""
    for i, chunk in enumerate(chunks):
        full_parts = [chunk.content]

        # Find all chunks that are nested under this one
        for j in range(i + 1, len(chunks)):
            other = chunks[j]
            # Check if other chunk is nested under this one by comparing paths
            if len(other.path) > len(chunk.path) and other.path[: len(chunk.path)] == chunk.path:
                if other.header:
                    full_parts.append(f"{'#' * other.level} {other.header}")
                full_parts.append(other.content)
            elif other.level <= chunk.level and other.level > 0:
                # Hit a sibling or parent header - stop
                break

        chunk.full_content = "\n\n".join(filter(None, full_parts))


def chunk_markdown_by_size(
    text: str,
    max_chars: int = 100000,
    overlap_chars: int = 500,
    respect_headers: bool = True,
) -> list[dict[str, str | int]]:
    """
    Chunk markdown by size while respecting header boundaries when possible.

    This is useful when you need chunks of roughly equal size but want to
    avoid splitting in the middle of sections.

    Args:
        text: The markdown text to chunk.
        max_chars: Maximum characters per chunk. Default 100K.
        overlap_chars: Number of characters to overlap between chunks for context.
        respect_headers: If True, try to split at header boundaries when possible.

    Returns:
        List of dicts with keys:
        - 'content': The chunk text
        - 'start_char': Starting character position
        - 'end_char': Ending character position
        - 'chunk_index': Index of this chunk

    Example:
        >>> chunks = chunk_markdown_by_size(long_text, max_chars=50000)
        >>> for chunk in chunks:
        ...     print(f"Chunk {chunk['chunk_index']}: {len(chunk['content'])} chars")
    """
    if len(text) <= max_chars:
        return [{"content": text, "start_char": 0, "end_char": len(text), "chunk_index": 0}]

    chunks = []
    current_pos = 0
    chunk_index = 0

    # Find all header positions for smart splitting
    header_pattern = re.compile(r"^#{1,6}\s+.+$", re.MULTILINE)
    header_positions = [m.start() for m in header_pattern.finditer(text)]
    header_positions.append(len(text))  # Add end position

    while current_pos < len(text):
        end_pos = min(current_pos + max_chars, len(text))

        if respect_headers and end_pos < len(text):
            # Find the best header boundary to split at
            best_split = end_pos

            # Look for a header position that's close to our target
            for hp in header_positions:
                if current_pos < hp <= end_pos:
                    best_split = hp
                elif hp > end_pos:
                    break

            # If we found a header boundary, use it (unless it's too early)
            if best_split > current_pos + (max_chars // 2):
                end_pos = best_split

        chunk_content = text[current_pos:end_pos]

        chunks.append(
            {
                "content": chunk_content,
                "start_char": current_pos,
                "end_char": end_pos,
                "chunk_index": chunk_index,
            }
        )

        # Move position, accounting for overlap
        current_pos = max(current_pos + 1, end_pos - overlap_chars)
        chunk_index += 1

    return chunks


def get_markdown_structure(text: str) -> str:
    """
    Get a quick overview of the markdown document structure (headers only).

    This is useful for understanding document organization before deciding
    on a chunking strategy.

    Args:
        text: The markdown text to analyze.

    Returns:
        A string showing the header hierarchy with indentation.

    Example:
        >>> structure = get_markdown_structure(text)
        >>> print(structure)
        # Chapter 1
          ## Section 1.1
          ## Section 1.2
            ### Subsection 1.2.1
        # Chapter 2
    """
    lines = text.split("\n")
    header_pattern = re.compile(r"^(#{1,6})\s+(.+)$")
    structure_lines = []

    for line in lines:
        match = header_pattern.match(line)
        if match:
            level = len(match.group(1))
            header_text = match.group(2).strip()
            indent = "  " * (level - 1)
            structure_lines.append(f"{indent}{'#' * level} {header_text}")

    return "\n".join(structure_lines)
