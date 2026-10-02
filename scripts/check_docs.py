#!/usr/bin/env python3
"""Check built Sphinx HTML for broken local assets and common math markup errors.

Usage: python scripts/check_docs.py docs/build/html
Uses only the Python standard library. This checks generated markup; a browser is
still needed to verify MathJax execution, external CDNs, and visual layout.
"""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit


VOID_TAGS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link",
    "meta", "param", "source", "track", "wbr",
}
SKIP_TEXT = {"head", "pre", "code", "script", "style", "textarea", "kbd", "samp"}
BLOCK_TAGS = {
    "address", "article", "blockquote", "dd", "div", "dl", "dt", "figcaption",
    "footer", "h1", "h2", "h3", "h4", "h5", "h6", "header", "li", "main",
    "nav", "ol", "p", "section", "table", "td", "th", "tr", "ul",
}
# A lone dollar left between parsed math nodes is another common MyST failure.
UNPARSED_MATH = re.compile(
    r"(?<!\\)\$\$|(?<!\\)\$[^\s$](?:[^$\n]*[^\s$])?(?<!\\)\$"
    r"|\\(?:begin|end)\{(?:equation|align|aligned|gather|multline|cases|[pbvBV]?matrix)\*?\}"
    r"|\\[\[(]"
)
BAD_MATH = re.compile(
    r"(?<!\\)\$|</?(?:font|span|div|p|u|strong|em)\s*>"
    r"|<(?:font|span|div|p|u|strong|em)\s+(?:color|class|style|id)\s*=|\*\*"
)
NESTED_EQUATION = re.compile(
    r"\\begin\{split\}[\s\S]*?\\begin\{(?:align|equation|gather|multline)\*?\}"
)


def local_target(root: Path, page: Path, url: str) -> tuple[Path | None, str | None]:
    """Resolve a local URL, checking exact filename case even on macOS."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return None, "invalid URL"
    if parts.scheme or parts.netloc or not parts.path:
        return None, None
    decoded = unquote(parts.path)
    base = root if decoded.startswith("/") else page.parent
    target = base / decoded.lstrip("/")
    # Walk instead of exists() alone: case-insensitive filesystems hide typos.
    current = Path(target.anchor)
    for name in target.parts[1:]:
        if name == "..":
            current = current.parent
            continue
        try:
            names = {child.name for child in current.iterdir()}
        except (OSError, ValueError):
            return target, "missing local target"
        if name not in names:
            reason = "filename case mismatch" if name.casefold() in {
                entry.casefold() for entry in names
            } else "missing local target"
            return target, reason
        current /= name
    if current.is_dir():
        current /= "index.html"
        if not current.is_file():
            return current, "directory has no index.html"
    return current, None


class PageChecker(HTMLParser):
    def __init__(self, root: Path, page: Path):
        super().__init__(convert_charrefs=True)
        self.root, self.page = root, page
        self.errors: list[tuple[int, str]] = []
        self.stack: list[tuple[str, bool, list[tuple[int, str]] | None]] = []
        self.prose: list[tuple[int, str]] = []
        self.math_count = 0
        self.first_math_line = 1
        self.has_mathjax = False
        self.resource_count = 0

    def error(self, line: int, message: str) -> None:
        issue = (line, message)
        if issue not in self.errors:
            self.errors.append(issue)

    def check_text(self, chunks: list[tuple[int, str]], *, math: bool = False) -> None:
        text = "".join(part for _, part in chunks)
        match = (BAD_MATH if math else UNPARSED_MATH).search(text)
        if math and not match:
            match = NESTED_EQUATION.search(text)
        lone_dollar = not math and text.strip() == "$"
        if not match and not lone_dollar:
            return
        offset = match.start() if match else text.index("$")
        line = chunks[0][0]
        for chunk_line, part in chunks:
            if offset < len(part):
                line = chunk_line + part[:offset].count("\n")
                break
            offset -= len(part)
        snippet = " ".join(text.strip().split())[:110]
        category = "malformed math (markup or nested equation)" if math else "unparsed math in page text"
        self.error(line, f"{category}: {snippet!r}")

    def flush_prose(self) -> None:
        self.check_text(self.prose)
        self.prose.clear()

    def check_resource(self, url: str, tag: str, attribute: str) -> None:
        target, reason = local_target(self.root, self.page, url)
        if target is not None:
            self.resource_count += 1
        if reason:
            self.error(self.getpos()[0], f"{reason}: <{tag} {attribute}={url!r}>")

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        for attribute in ("href", "src", "poster"):
            if values.get(attribute):
                self.check_resource(values[attribute], tag, attribute)
        if tag == "script":
            src = values.get("src", "") or ""
            if "mathjax" in src.lower() or re.search(
                r"(?:^|/)tex-(?:mml-)?(?:chtml|svg)\.js(?:[?#]|$)", src
            ):
                self.has_mathjax = True
        skipped = tag in SKIP_TEXT or any(skip for _, skip, _ in self.stack)
        is_math = "math" in (values.get("class", "") or "").split() and not skipped
        if tag in BLOCK_TAGS or tag in SKIP_TEXT or is_math:
            self.flush_prose()
        math = [] if is_math else None
        if is_math:
            self.math_count += 1
            if self.math_count == 1:
                self.first_math_line = self.getpos()[0]
        if tag not in VOID_TAGS:
            self.stack.append((tag, skipped, math))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in VOID_TAGS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in BLOCK_TAGS or tag in SKIP_TEXT:
            self.flush_prose()
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                for _, _, math in self.stack[index:]:
                    if math is not None:
                        self.check_text(math, math=True)
                del self.stack[index:]
                break

    def handle_data(self, data: str) -> None:
        if any(skip for _, skip, _ in self.stack):
            return
        for _, _, math in reversed(self.stack):
            if math is not None:
                math.append((self.getpos()[0], data))
                return
        self.prose.append((self.getpos()[0], data))

    def finish(self) -> None:
        self.close()
        self.flush_prose()
        if self.math_count and not self.has_mathjax:
            self.error(self.first_math_line, "page contains math but has no MathJax script")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html_directory", type=Path)
    root = parser.parse_args().html_directory.resolve()
    pages = sorted(root.rglob("*.html")) if root.is_dir() else []
    if not pages:
        print(f"No HTML pages found in {root}", file=sys.stderr)
        return 1
    errors = math_count = resource_count = 0
    for page in pages:
        checker = PageChecker(root, page)
        try:
            checker.feed(page.read_text(encoding="utf-8"))
            checker.finish()
        except (OSError, UnicodeError) as exc:
            checker.error(1, f"cannot read HTML: {exc}")
        for line, message in checker.errors:
            print(f"{page.relative_to(root)}:{line}: {message}", file=sys.stderr)
        errors += len(checker.errors)
        math_count += checker.math_count
        resource_count += checker.resource_count
    print(f"Checked {len(pages)} HTML pages, {math_count} math nodes, "
          f"{resource_count} local references: {errors} error(s).")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
