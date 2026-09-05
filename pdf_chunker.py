from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

try:
    import fitz  # PyMuPDF
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyMuPDF is required. Install it with: pip install pymupdf"
    ) from exc

try:  # optional, but gives accurate token counts for OpenAI-style embeddings
    import tiktoken
except ImportError:  # pragma: no cover
    tiktoken = None


DEFAULT_PDFS = [
    r"C:\Users\MSI\Desktop\rag_chatbot\SOP-50-10-7.1-effective-11.15.pdf",
    r"C:\Users\MSI\Desktop\rag_chatbot\p334.pdf",
]


# -----------------------------
# Data models

# -----------------------------


@dataclass
class RawLine:
    page: int
    block_id: int
    line_id: int
    text: str
    bbox: tuple[float, float, float, float]
    font_size: float
    bold: bool
    italic: bool


@dataclass
class TableElement:
    page: int
    bbox: tuple[float, float, float, float]
    markdown: str


@dataclass
class Element:
    kind: str  # heading | paragraph | list | table | note
    text: str
    page_start: int
    page_end: int
    heading_level: Optional[int] = None
    source_bbox: Optional[tuple[float, float, float, float]] = None


@dataclass
class Section:
    document_id: str
    title: str
    section_path: list[str]
    heading_level: int
    elements: list[Element] = field(default_factory=list)


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    document_title: str
    document_type: str
    authority: str
    jurisdiction: str
    version: str
    source_file: str
    topic_id: str
    section_title: str
    section_path: list[str]
    breadcrumb: str
    chunk_index: int
    chunk_index_in_topic: int
    page_start: int
    page_end: int
    token_count: int
    overlap_from_previous_tokens: int
    previous_chunk_id: Optional[str]
    next_chunk_id: Optional[str]
    previous_in_topic_id: Optional[str]
    next_in_topic_id: Optional[str]
    tags: list[str]
    content: str
    embedding_text: str


@dataclass(frozen=True)
class DocumentProfile:
    document_id: str
    title: str
    document_type: str
    authority: str
    jurisdiction: str
    version: str
    tags: tuple[str, ...]
    profile: str


@dataclass
class PageData:
    page_number: int
    width: float
    height: float
    lines: list[RawLine]
    tables: list[TableElement]


# -----------------------------
# Token counting
# -----------------------------


class TokenCounter:
    """Accurate with tiktoken when installed; safe approximation otherwise."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self.encoding = None
        if tiktoken is not None:
            try:
                self.encoding = tiktoken.get_encoding(encoding_name)
            except Exception:
                self.encoding = None

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self.encoding is not None:
            return len(self.encoding.encode(text, disallowed_special=()))
        # English legal/tax prose averages ~4 chars/token. This fallback errs slightly high.
        return max(1, int(len(text) / 3.7))


# -----------------------------
# Document profiling
# -----------------------------


def detect_profile(path: Path, first_pages_text: str) -> DocumentProfile:
    probe = re.sub(r"\s+", " ", (path.name + "\n" + first_pages_text[:12000]).lower())

    if "sop 50 10" in probe and "small business administration" in probe:
        return DocumentProfile(
            document_id="sba_sop_50_10_7_1",
            title="SBA SOP 50 10 7.1 — Lender and Development Company Loan Programs",
            document_type="loan_policy",
            authority="U.S. Small Business Administration (SBA)",
            jurisdiction="US federal",
            version="SOP 50 10 7.1; effective 2023-11-15",
            tags=("sba", "loan", "7(a)", "504", "small_business", "eligibility", "lending"),
            profile="sba_sop",
        )

    if "publication 334" in probe and "tax guide for small business" in probe:
        return DocumentProfile(
            document_id="irs_pub_334_2025",
            title="IRS Publication 334 (2025) — Tax Guide for Small Business",
            document_type="tax_guide",
            authority="Internal Revenue Service (IRS)",
            jurisdiction="US federal",
            version="Publication 334 for 2025 returns; published 2026-02-10",
            tags=("irs", "tax", "schedule_c", "self_employed", "small_business", "2025_returns"),
            profile="irs_pub334",
        )

    stem = slugify(path.stem)
    return DocumentProfile(
        document_id=stem or "document",
        title=path.stem,
        document_type="reference",
        authority="Unknown",
        jurisdiction="Unknown",
        version="Unknown",
        tags=("reference",),
        profile="generic",
    )


# -----------------------------
# Text / layout helpers
# -----------------------------


def clean_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = text.replace("\ufeff", "")
    text = text.replace("\ufffe", "")
    text = text.replace("\u2060", "")
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def join_spans(spans: Sequence[dict]) -> str:
    if not spans:
        return ""
    pieces: list[str] = []
    for span in spans:
        txt = clean_text(str(span.get("text", "")))
        if not txt:
            continue
        if not pieces:
            pieces.append(txt)
            continue
        prev = pieces[-1]
        # Avoid "Form W- 9" and similar artifacts; otherwise use a normal space.
        if prev.endswith(("/", "-", "(", "$")) or txt.startswith((")", ",", ".", ";", ":", "%")):
            pieces[-1] = prev + txt
        else:
            pieces.append(txt)
    return " ".join(pieces)


def normalized(text: str) -> str:
    text = clean_text(text).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"\bchapter\s+\d+[a-z]?\b", " ", text)
    text = re.sub(r"\bsection\s+([a-z])\b", r" section \1 ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def slugify(text: str, max_len: int = 80) -> str:
    value = normalized(text).replace(" ", "-")
    value = re.sub(r"-+", "-", value).strip("-")
    return value[:max_len]


def stable_id(*parts: str, size: int = 12) -> str:
    joined = "||".join(parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:size]


def bbox_inside(inner: tuple[float, float, float, float], outer: tuple[float, float, float, float]) -> bool:
    ix0, iy0, ix1, iy1 = inner
    ox0, oy0, ox1, oy1 = outer
    cx = (ix0 + ix1) / 2
    cy = (iy0 + iy1) / 2
    return ox0 <= cx <= ox1 and oy0 <= cy <= oy1


def line_is_bold(spans: Sequence[dict]) -> bool:
    chars = 0
    bold_chars = 0
    for span in spans:
        txt = str(span.get("text", ""))
        n = max(1, len(txt.strip()))
        chars += n
        font = str(span.get("font", "")).lower()
        flags = int(span.get("flags", 0))
        if "bold" in font or flags & 16:
            bold_chars += n
    return chars > 0 and bold_chars / chars >= 0.55


def line_is_italic(spans: Sequence[dict]) -> bool:
    chars = 0
    italic_chars = 0
    for span in spans:
        txt = str(span.get("text", ""))
        n = max(1, len(txt.strip()))
        chars += n
        font = str(span.get("font", "")).lower()
        flags = int(span.get("flags", 0))
        if "italic" in font or flags & 2:
            italic_chars += n
    return chars > 0 and italic_chars / chars >= 0.55


def markdown_table(rows: Sequence[Sequence[object]]) -> str:
    cleaned: list[list[str]] = []
    max_cols = 0
    for row in rows:
        vals = [clean_text("" if cell is None else str(cell)).replace("\n", " ") for cell in row]
        max_cols = max(max_cols, len(vals))
        cleaned.append(vals)
    if not cleaned or max_cols == 0:
        return ""
    for row in cleaned:
        row.extend([""] * (max_cols - len(row)))

    def esc(s: str) -> str:
        return s.replace("|", "\\|")

    header = cleaned[0]
    body = cleaned[1:]
    lines = [
        "| " + " | ".join(esc(x) for x in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(esc(x) for x in row) + " |")
    return "\n".join(lines)


# -----------------------------
# PDF extraction
# -----------------------------


def extract_pages(doc: fitz.Document, table_mode: str = "auto") -> list[PageData]:
    pages: list[PageData] = []
    flags = fitz.TEXTFLAGS_DICT | getattr(fitz, "TEXT_DEHYPHENATE", 0)

    for pno in range(doc.page_count):
        page = doc.load_page(pno)
        page_dict = page.get_text("dict", sort=False, flags=flags)
        raw_lines: list[RawLine] = []

        for block_id, block in enumerate(page_dict.get("blocks", [])):
            if block.get("type") != 0:
                continue
            for line_id, line in enumerate(block.get("lines", [])):
                spans = line.get("spans", [])
                text = join_spans(spans)
                if not text:
                    continue
                sizes = [float(s.get("size", 0.0)) for s in spans if str(s.get("text", "")).strip()]
                font_size = max(sizes) if sizes else 0.0
                raw_lines.append(
                    RawLine(
                        page=pno + 1,
                        block_id=block_id,
                        line_id=line_id,
                        text=text,
                        bbox=tuple(float(x) for x in line.get("bbox", (0, 0, 0, 0))),
                        font_size=font_size,
                        bold=line_is_bold(spans),
                        italic=line_is_italic(spans),
                    )
                )

        tables: list[TableElement] = []
        page_text_probe = " ".join(ln.text for ln in raw_lines).lower()
        likely_table = bool(
            re.search(r"\btable\s+[a-z0-9][a-z0-9-]*", page_text_probe, re.I)
            or ("if you" in page_text_probe and "then" in page_text_probe)
            or ("which forms must i file" in page_text_probe)
        )
        do_tables = table_mode == "all" or (table_mode == "auto" and likely_table)
        if do_tables and hasattr(page, "find_tables"):
            try:
                found = page.find_tables()
                for table in getattr(found, "tables", []):
                    rows = table.extract()
                    md = markdown_table(rows)
                    if md and len(rows) >= 2:
                        tables.append(
                            TableElement(
                                page=pno + 1,
                                bbox=tuple(float(x) for x in table.bbox),
                                markdown=md,
                            )
                        )
            except Exception:
                # Table extraction is an enhancement, never a reason to lose the page.
                tables = []

        pages.append(
            PageData(
                page_number=pno + 1,
                width=float(page.rect.width),
                height=float(page.rect.height),
                lines=raw_lines,
                tables=tables,
            )
        )
    return pages


def boilerplate_signatures(pages: Sequence[PageData]) -> set[str]:
    counts: Counter[str] = Counter()
    total_pages = max(1, len(pages))
    for page in pages:
        for line in page.lines:
            _, y0, _, y1 = line.bbox
            in_margin = y1 <= page.height * 0.10 or y0 >= page.height * 0.90
            if not in_margin:
                continue
            sig = normalized(re.sub(r"\d+", "#", line.text))
            if 4 <= len(sig) <= 140:
                counts[sig] += 1
    threshold = max(4, int(total_pages * 0.04))
    return {sig for sig, n in counts.items() if n >= threshold}


def is_boilerplate_line(
    line: RawLine,
    page: PageData,
    profile: DocumentProfile,
    repeated_signatures: set[str],
) -> bool:
    text = clean_text(line.text)
    low = text.lower()
    _, y0, _, y1 = line.bbox
    top = y1 <= page.height * 0.12
    bottom = y0 >= page.height * 0.88

    if re.fullmatch(r"\d+", text) and (top or bottom):
        return True

    sig = normalized(re.sub(r"\d+", "#", text))
    if (top or bottom) and sig in repeated_signatures:
        return True

    if profile.profile == "irs_pub334":
        if top and (low.startswith("page ") and " of " in low and "fileid" in low):
            return True
        if top and "the type and rule above prints on all proofs" in low:
            return True
        if bottom and ("publication 334 (2025)" in low or low.startswith("feb 10, 2026")):
            return True

    if profile.profile == "sba_sop":
        if y1 <= 72 and not line.bold:
            return True
        if top and re.match(r"^sop\s+50\s+10\s+7(?:\.1)?\b", low):
            return True
        if bottom and "effective november 15, 2023" in low and "page" in low:
            return True

    return False


def should_skip_page(page: PageData, profile: DocumentProfile) -> bool:
    text = " ".join(line.text for line in page.lines[:80])
    low = text.lower()

    # Retrieval pollution: contents and index pages are navigation aids, not source rules.
    if profile.profile == "sba_sop" and page.page_number in {1, 2, 3, 4, 5, 7, 8}:
        return True
    if profile.profile == "irs_pub334" and page.page_number == 1:
        return True
    if profile.profile == "irs_pub334" and page.page_number >= 54:
        return True
    return False


# -----------------------------
# Heading detection from PDF outline + visual fallback
# -----------------------------


def outline_by_page(doc: fitz.Document) -> dict[int, list[tuple[int, str]]]:
    result: dict[int, list[tuple[int, str]]] = defaultdict(list)
    try:
        for level, title, page_no in doc.get_toc(simple=True):
            if page_no and title:
                result[int(page_no)].append((int(level), clean_text(title)))
    except Exception:
        pass
    return result


def match_score(candidate: str, target: str) -> float:
    c = normalized(candidate)
    t = normalized(target)
    if not c or not t:
        return 0.0
    # Ignore chapter numbering discrepancy between bookmarks and printed page title.
    c2 = re.sub(r"^(?:\d+|[a-z]|[ivxlcdm]+)\s+", "", c)
    t2 = re.sub(r"^(?:\d+|[a-z]|[ivxlcdm]+)\s+", "", t)
    if c2 == t2:
        return 1.0
    if len(c2) >= 5 and (c2 in t2 or t2 in c2):
        ratio = min(len(c2), len(t2)) / max(len(c2), len(t2))
        if ratio >= 0.70:
            return 0.96
    return SequenceMatcher(None, c2, t2).ratio()


def normalize_heading_level(profile: DocumentProfile, title: str, toc_level: int) -> int:
    low = title.lower().strip()
    if profile.profile == "sba_sop":
        if low.startswith("section "):
            return 1
        if low.startswith("chapter "):
            return 2
        if low.startswith("appendix "):
            return 2
        if low.startswith("sba 7(a) and 504 business loan requirements"):
            return 1
        if low in {"appendices", "table of contents", "user tips: how to use this document"}:
            return 1
        return 3

    if profile.profile == "irs_pub334":
        if re.match(r"^chapter\s+\d+", low):
            return 1
        return max(1, min(4, toc_level))

    return max(1, min(4, toc_level))


def find_outline_heading_ranges(
    lines: Sequence[RawLine],
    entries: Sequence[tuple[int, str]],
    profile: DocumentProfile,
) -> dict[int, tuple[int, int, str, int]]:
    """Map start-line index -> (start, end_exclusive, title, normalized_level)."""
    matches: dict[int, tuple[int, int, str, int]] = {}
    cursor = 0

    for toc_level, title in entries:
        best: Optional[tuple[float, int, int]] = None
        max_window = 5
        for i in range(cursor, len(lines)):
            # A bookmark title almost always appears within a few printed lines.
            for width in range(1, min(max_window, len(lines) - i) + 1):
                candidate = " ".join(lines[j].text for j in range(i, i + width))
                score = match_score(candidate, title)
                if best is None or score > best[0]:
                    best = (score, i, i + width)
                if score >= 0.995:
                    break
            if best is not None and best[0] >= 0.995:
                # Stop only on an essentially exact match; a merely strong fuzzy
                # match can otherwise consume the final line of the previous paragraph.
                break

        if best is not None and best[0] >= 0.84:
            _, start, end = best
            level = normalize_heading_level(profile, title, toc_level)
            matches[start] = (start, end, title, level)
            cursor = end

    return matches


def visual_heading(line: RawLine, body_font: float, profile: DocumentProfile) -> Optional[int]:
    text = clean_text(line.text)
    if not text or len(text) > 180 or not re.search(r"[A-Za-z0-9]", text):
        return None
    if NOTE_ONLY_RE.match(text):
        return None
    low = text.lower()

    if profile.profile == "sba_sop":
        # The SBA PDF has many bold lead-in sentences. Treat only true all-caps
        # policy headings as visual fallbacks; primary headings are taken from bookmarks.
        if text.isupper() and line.bold and len(text.split()) <= 22:
            if re.match(r"^SECTION\s+[A-Z][\.:]", text):
                return 1
            if re.match(r"^CHAPTER\s+\d+[\.:]", text):
                return 2
            return 3

    if profile.profile == "irs_pub334":
        # IRS chapter and section titles have a strong typographic hierarchy.
        if line.bold and line.font_size >= max(14.0, body_font + 4.0):
            if re.fullmatch(r"\d+\.", text):
                return None
            return 1 if line.font_size >= 18.0 else 2

    if line.bold and line.font_size >= body_font + 3.0 and len(text.split()) <= 18:
        return 2
    return None


def body_font_size(pages: Sequence[PageData]) -> float:
    sizes: list[float] = []
    for page in pages[: min(60, len(pages))]:
        for line in page.lines:
            if len(line.text) >= 25 and 6.0 <= line.font_size <= 16.0 and not line.bold:
                sizes.append(round(line.font_size, 1))
    return statistics.median(sizes) if sizes else 10.0


# -----------------------------
# Page -> semantic elements
# -----------------------------


BULLET_RE = re.compile(r"^(?:[•●▪◦]|[-–—]\s+|\(?\d+[.)]\s+|[a-zA-Z][.)]\s+|[ivxlcdm]+[.)]\s+)")
NOTE_ONLY_RE = re.compile(r"^(?:CAUTION|TIP|NOTE|IMPORTANT)[:!]*$", re.I)


def merge_wrapped_lines(lines: Sequence[str]) -> str:
    out = ""
    for raw in lines:
        s = clean_text(raw)
        if not s:
            continue
        if not out:
            out = s
            continue
        # PyMuPDF TEXT_DEHYPHENATE already resolves most discretionary hyphenation.
        # Do not remove literal hyphens here because legal terms such as "short-term" matter.
        if out.endswith("-"):
            out += s
        else:
            out += " " + s
    return re.sub(r"\s+", " ", out).strip()


def page_to_elements(
    page: PageData,
    entries: Sequence[tuple[int, str]],
    profile: DocumentProfile,
    repeated_signatures: set[str],
    body_font: float,
) -> list[Element]:
    if should_skip_page(page, profile):
        return []

    table_bboxes = [t.bbox for t in page.tables]
    lines = [
        ln
        for ln in page.lines
        if not is_boilerplate_line(ln, page, profile, repeated_signatures)
        and not any(bbox_inside(ln.bbox, tb) for tb in table_bboxes)
    ]

    outline_ranges = find_outline_heading_ranges(lines, entries, profile)
    consumed_heading_indexes: set[int] = set()
    for start, end, _, _ in outline_ranges.values():
        consumed_heading_indexes.update(range(start, end))

    # Insert tables by vertical position. Most tables in these two references are full-width.
    table_items = sorted(page.tables, key=lambda t: (t.bbox[1], t.bbox[0]))
    table_cursor = 0

    elements: list[Element] = []
    pending_lines: list[RawLine] = []
    pending_label: Optional[str] = None
    current_block: Optional[int] = None

    def flush_pending() -> None:
        nonlocal pending_lines, current_block, pending_label
        if not pending_lines:
            return
        text = merge_wrapped_lines([x.text for x in pending_lines])
        if pending_label:
            text = f"{pending_label}: {text}" if text else pending_label
            pending_label = None
        if text:
            kind = "list" if BULLET_RE.match(text) else "paragraph"
            elements.append(
                Element(
                    kind=kind,
                    text=text,
                    page_start=page.page_number,
                    page_end=page.page_number,
                    source_bbox=(
                        min(x.bbox[0] for x in pending_lines),
                        min(x.bbox[1] for x in pending_lines),
                        max(x.bbox[2] for x in pending_lines),
                        max(x.bbox[3] for x in pending_lines),
                    ),
                )
            )
        pending_lines = []
        current_block = None

    i = 0
    while i < len(lines):
        line = lines[i]

        # Emit tables that occur above this line.
        while table_cursor < len(table_items) and table_items[table_cursor].bbox[1] <= line.bbox[1]:
            flush_pending()
            table = table_items[table_cursor]
            elements.append(
                Element(
                    kind="table",
                    text=table.markdown,
                    page_start=page.page_number,
                    page_end=page.page_number,
                    source_bbox=table.bbox,
                )
            )
            table_cursor += 1

        if i in outline_ranges:
            flush_pending()
            start, end, title, level = outline_ranges[i]
            elements.append(
                Element(
                    kind="heading",
                    text=title,
                    page_start=page.page_number,
                    page_end=page.page_number,
                    heading_level=level,
                    source_bbox=line.bbox,
                )
            )
            i = end
            continue

        if i in consumed_heading_indexes:
            i += 1
            continue

        maybe_level = visual_heading(line, body_font, profile)
        if maybe_level is not None:
            flush_pending()
            elements.append(
                Element(
                    kind="heading",
                    text=line.text,
                    page_start=page.page_number,
                    page_end=page.page_number,
                    heading_level=maybe_level,
                    source_bbox=line.bbox,
                )
            )
            i += 1
            continue

        if NOTE_ONLY_RE.match(line.text):
            flush_pending()
            pending_label = clean_text(line.text).rstrip(":!")
            i += 1
            continue

        # A new explicit bullet/list marker starts a new semantic unit.
        if BULLET_RE.match(line.text) and pending_lines:
            flush_pending()

        # Respect the PDF's native block order. It is materially better than y/x sorting
        # for the two-column IRS publication.
        if current_block is not None and line.block_id != current_block:
            flush_pending()

        pending_lines.append(line)
        current_block = line.block_id
        i += 1

    flush_pending()

    while table_cursor < len(table_items):
        table = table_items[table_cursor]
        elements.append(
            Element(
                kind="table",
                text=table.markdown,
                page_start=page.page_number,
                page_end=page.page_number,
                source_bbox=table.bbox,
            )
        )
        table_cursor += 1

    return elements


# -----------------------------
# Element stream -> sections
# -----------------------------


def build_sections(profile: DocumentProfile, elements: Sequence[Element]) -> list[Section]:
    sections: list[Section] = []
    stack: list[str] = []
    current: Optional[Section] = None

    def ensure_preamble() -> Section:
        nonlocal current
        if current is None:
            current = Section(
                document_id=profile.document_id,
                title="Document introduction",
                section_path=["Document introduction"],
                heading_level=1,
            )
            sections.append(current)
        return current

    for el in elements:
        if el.kind == "heading":
            level = max(1, min(6, int(el.heading_level or 2)))
            while len(stack) >= level:
                stack.pop()
            while len(stack) < level - 1:
                stack.append("Context")
            stack.append(clean_text(el.text))
            current = Section(
                document_id=profile.document_id,
                title=clean_text(el.text),
                section_path=list(stack),
                heading_level=level,
            )
            sections.append(current)
        else:
            ensure_preamble().elements.append(el)

    return [s for s in sections if s.elements]


def section_is_retrieval_noise(profile: DocumentProfile, section: Section) -> bool:
    title = normalized(section.title)
    if title in {"contents", "table of contents", "index"}:
        return True
    if profile.profile == "sba_sop" and title.startswith("user tips how to use this document"):
        return True
    if profile.profile == "irs_pub334" and title == "photographs of missing children":
        return True
    return False


# -----------------------------
# Chunking
# -----------------------------


def sentence_split(text: str) -> list[str]:
    # Conservative: keep statutory abbreviations / form numbers intact when possible.
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9“\"(])", text)
    return [p.strip() for p in parts if p.strip()]


def split_long_text(text: str, counter: TokenCounter, max_tokens: int) -> list[str]:
    if counter.count(text) <= max_tokens:
        return [text]
    sentences = sentence_split(text)
    if len(sentences) <= 1:
        # Last-resort character window; keeps the module dependency-light.
        approx_chars = max(500, int(max_tokens * 3.4))
        return [text[i : i + approx_chars].strip() for i in range(0, len(text), approx_chars) if text[i : i + approx_chars].strip()]

    parts: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        trial = " ".join(current + [sentence])
        if current and counter.count(trial) > max_tokens:
            parts.append(" ".join(current))
            current = [sentence]
        else:
            current.append(sentence)
    if current:
        parts.append(" ".join(current))
    return parts


def split_large_markdown_table(table_md: str, counter: TokenCounter, max_tokens: int) -> list[str]:
    if counter.count(table_md) <= max_tokens:
        return [table_md]
    lines = [ln for ln in table_md.splitlines() if ln.strip()]
    if len(lines) < 4:
        return split_long_text(table_md, counter, max_tokens)
    header = lines[:2]
    rows = lines[2:]
    parts: list[str] = []
    current_rows: list[str] = []
    for row in rows:
        trial = "\n".join(header + current_rows + [row])
        if current_rows and counter.count(trial) > max_tokens:
            parts.append("\n".join(header + current_rows))
            current_rows = [row]
        else:
            current_rows.append(row)
    if current_rows:
        parts.append("\n".join(header + current_rows))
    return parts


def expand_elements_for_limits(
    elements: Sequence[Element], counter: TokenCounter, max_tokens: int
) -> list[Element]:
    expanded: list[Element] = []
    for el in elements:
        if counter.count(el.text) <= max_tokens:
            expanded.append(el)
            continue
        if el.kind == "table":
            pieces = split_large_markdown_table(el.text, counter, max_tokens)
        else:
            pieces = split_long_text(el.text, counter, max_tokens)
        for piece in pieces:
            expanded.append(
                Element(
                    kind=el.kind,
                    text=piece,
                    page_start=el.page_start,
                    page_end=el.page_end,
                    source_bbox=el.source_bbox,
                )
            )
    return expanded


def semantic_overlap(
    units: Sequence[Element], counter: TokenCounter, overlap_tokens: int
) -> list[Element]:
    if overlap_tokens <= 0:
        return []
    selected: list[Element] = []
    total = 0
    for unit in reversed(units):
        n = counter.count(unit.text)
        # Never let overlap become a second full chunk.
        if selected and total + n > overlap_tokens * 1.35:
            break
        selected.append(unit)
        total += n
        if total >= overlap_tokens:
            break
    return list(reversed(selected))


def render_units(units: Sequence[Element]) -> str:
    return "\n\n".join(u.text.strip() for u in units if u.text.strip()).strip()


def build_embedding_text(profile: DocumentProfile, section: Section, content: str, page_start: int, page_end: int) -> str:
    breadcrumb = " > ".join(section.section_path)
    page_label = str(page_start) if page_start == page_end else f"{page_start}-{page_end}"
    return (
        f"Document: {profile.title}\n"
        f"Authority: {profile.authority}\n"
        f"Version: {profile.version}\n"
        f"Topic: {breadcrumb}\n"
        f"Source pages: {page_label}\n\n"
        f"{content}"
    )


def chunk_sections(
    profile: DocumentProfile,
    source_file: str,
    sections: Sequence[Section],
    max_tokens: int = 750,
    overlap_tokens: int = 110,
    min_tokens: int = 80,
    counter: Optional[TokenCounter] = None,
) -> list[Chunk]:
    counter = counter or TokenCounter()
    chunks: list[Chunk] = []
    global_index = 0

    for section_ordinal, section in enumerate(sections):
        if section_is_retrieval_noise(profile, section):
            continue

        # Reserve space for breadcrumb / authority / version metadata that is prepended
        # to embedding_text so max_tokens applies to the actual embedded record.
        prefix_probe = build_embedding_text(profile, section, "", 1, 1)
        content_budget = max(200, max_tokens - counter.count(prefix_probe) - 8)
        units = expand_elements_for_limits(section.elements, counter, max_tokens=content_budget)
        if not units:
            continue

        first_section_page = min((e.page_start for e in section.elements), default=0)
        topic_key = (
            profile.document_id
            + "||"
            + " > ".join(section.section_path)
            + f"||section:{section_ordinal}||page:{first_section_page}"
        )
        topic_id = f"topic_{stable_id(topic_key)}"
        topic_chunks: list[tuple[list[Element], int]] = []
        current: list[Element] = []
        overlap_count = 0

        for unit in units:
            trial = render_units(current + [unit])
            if current and counter.count(trial) > content_budget:
                topic_chunks.append((list(current), overlap_count))
                carry = semantic_overlap(current, counter, overlap_tokens)
                overlap_count = counter.count(render_units(carry))
                current = carry + [unit]
                # If carry + a large atomic unit still exceeds the cap, drop carry.
                if counter.count(render_units(current)) > content_budget:
                    current = [unit]
                    overlap_count = 0
            else:
                current.append(unit)

        if current:
            topic_chunks.append((list(current), overlap_count))

        # Merge an extremely small final chunk back when this is safe.
        if len(topic_chunks) >= 2:
            last_units, last_overlap = topic_chunks[-1]
            last_text = render_units(last_units)
            prev_units, prev_overlap = topic_chunks[-2]
            prev_text = render_units(prev_units)
            if counter.count(last_text) < min_tokens and counter.count(prev_text + "\n\n" + last_text) <= content_budget:
                topic_chunks[-2] = (prev_units + last_units, prev_overlap)
                topic_chunks.pop()

        temp_ids: list[str] = []
        for topic_idx, (chunk_units, _) in enumerate(topic_chunks):
            temp_ids.append(f"{profile.document_id}:{topic_id}:{topic_idx:04d}")

        for topic_idx, (chunk_units, overlap_count) in enumerate(topic_chunks):
            content = render_units(chunk_units)
            if not content:
                continue
            page_start = min(u.page_start for u in chunk_units)
            page_end = max(u.page_end for u in chunk_units)
            embedding_text = build_embedding_text(profile, section, content, page_start, page_end)
            chunk_id = temp_ids[topic_idx]

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=profile.document_id,
                    document_title=profile.title,
                    document_type=profile.document_type,
                    authority=profile.authority,
                    jurisdiction=profile.jurisdiction,
                    version=profile.version,
                    source_file=source_file,
                    topic_id=topic_id,
                    section_title=section.title,
                    section_path=section.section_path,
                    breadcrumb=" > ".join(section.section_path),
                    chunk_index=global_index,
                    chunk_index_in_topic=topic_idx,
                    page_start=page_start,
                    page_end=page_end,
                    token_count=counter.count(embedding_text),
                    overlap_from_previous_tokens=overlap_count,
                    previous_chunk_id=None,  # filled after all documents are assembled
                    next_chunk_id=None,
                    previous_in_topic_id=temp_ids[topic_idx - 1] if topic_idx > 0 else None,
                    next_in_topic_id=temp_ids[topic_idx + 1] if topic_idx + 1 < len(temp_ids) else None,
                    tags=list(profile.tags),
                    content=content,
                    embedding_text=embedding_text,
                )
            )
            global_index += 1

    return chunks


def link_global_neighbors(chunks: list[Chunk]) -> None:
    for i, chunk in enumerate(chunks):
        prev = chunks[i - 1] if i > 0 else None
        nxt = chunks[i + 1] if i + 1 < len(chunks) else None
        # Never imply continuity across different source documents.
        chunk.previous_chunk_id = prev.chunk_id if prev and prev.document_id == chunk.document_id else None
        chunk.next_chunk_id = nxt.chunk_id if nxt and nxt.document_id == chunk.document_id else None
        chunk.chunk_index = i


# -----------------------------
# End-to-end document processing
# -----------------------------


def process_pdf(
    path: Path,
    max_tokens: int,
    overlap_tokens: int,
    min_tokens: int,
    table_mode: str,
    emit_markdown_dir: Optional[Path] = None,
) -> tuple[list[Chunk], dict]:
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    doc = fitz.open(path)
    try:
        first_pages_text = "\n".join(doc[i].get_text("text", sort=False) for i in range(min(3, doc.page_count)))
        profile = detect_profile(path, first_pages_text)
        pages = extract_pages(doc, table_mode=table_mode)
        repeats = boilerplate_signatures(pages)
        toc = outline_by_page(doc)
        base_font = body_font_size(pages)

        all_elements: list[Element] = []
        for page in pages:
            all_elements.extend(
                page_to_elements(
                    page=page,
                    entries=toc.get(page.page_number, []),
                    profile=profile,
                    repeated_signatures=repeats,
                    body_font=base_font,
                )
            )

        sections = build_sections(profile, all_elements)
        counter = TokenCounter()
        chunks = chunk_sections(
            profile=profile,
            source_file=path.name,
            sections=sections,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            min_tokens=min_tokens,
            counter=counter,
        )

        if emit_markdown_dir is not None:
            emit_markdown_dir.mkdir(parents=True, exist_ok=True)
            md_path = emit_markdown_dir / f"{path.stem}.parsed.md"
            with md_path.open("w", encoding="utf-8") as f:
                f.write(f"# {profile.title}\n\n")
                f.write(f"> {profile.version}\n\n")
                current_page = None
                for el in all_elements:
                    if current_page != el.page_start:
                        current_page = el.page_start
                        f.write(f"\n<!-- page: {current_page} -->\n\n")
                    if el.kind == "heading":
                        level = max(1, min(6, el.heading_level or 2))
                        f.write("#" * level + " " + el.text + "\n\n")
                    else:
                        f.write(el.text + "\n\n")

        stats = {
            "document_id": profile.document_id,
            "source_file": path.name,
            "pages": doc.page_count,
            "outline_entries": sum(len(v) for v in toc.values()),
            "sections_with_content": len(sections),
            "chunks": len(chunks),
            "body_font_size": base_font,
            "tables_detected": sum(len(p.tables) for p in pages),
            "profile": profile.profile,
            "version": profile.version,
        }
        return chunks, stats
    finally:
        doc.close()


# -----------------------------
# Output / CLI
# -----------------------------


def write_jsonl(path: Path, chunks: Sequence[Chunk]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def write_json(path: Path, chunks: Sequence[Chunk]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in chunks], f, ensure_ascii=False, indent=2)


def write_manifest(path: Path, stats: Sequence[dict], chunks: Sequence[Chunk], args: argparse.Namespace) -> None:
    token_counts = [c.token_count for c in chunks]
    manifest = {
        "documents": list(stats),
        "total_chunks": len(chunks),
        "chunking": {
            "max_tokens": args.max_tokens,
            "overlap_tokens": args.overlap_tokens,
            "min_tokens": args.min_tokens,
            "table_extraction": args.tables,
            "tokenizer": "tiktoken/cl100k_base" if tiktoken is not None else "character approximation",
            "strategy": "PDF-outline + layout-aware hierarchical semantic chunking; overlap only within topic",
        },
        "token_stats": {
            "min": min(token_counts) if token_counts else 0,
            "max": max(token_counts) if token_counts else 0,
            "mean": round(sum(token_counts) / len(token_counts), 1) if token_counts else 0,
        },
    }
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Structure-aware PDF chunker for the SBA SOP and IRS Publication 334 RAG corpus."
    )
    parser.add_argument(
        "--input",
        action="append",
        dest="inputs",
        help="PDF path. Repeat --input for multiple PDFs. Defaults to the two project PDFs.",
    )
    parser.add_argument("--output", default="data/chunks.jsonl", help="Output .jsonl or .json file.")
    parser.add_argument("--max-tokens", type=int, default=800, help="Maximum tokens in embedding_text, including metadata prefix.")
    parser.add_argument("--overlap-tokens", type=int, default=100, help="Semantic overlap within the same topic.")
    parser.add_argument("--min-tokens", type=int, default=80, help="Tiny tail chunks below this may be merged backward.")
    parser.add_argument(
        "--tables",
        choices=("auto", "all", "off"),
        default="auto",
        help="Table extraction: auto=only likely table pages (recommended), all=every page, off=disabled.",
    )
    parser.add_argument(
        "--emit-markdown",
        action="store_true",
        help="Also write parsed Markdown files for inspection/debugging. Markdown is not required for chunking.",
    )
    parser.add_argument("--markdown-dir", default="data/parsed", help="Directory for --emit-markdown output.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    input_paths = [Path(p) for p in (args.inputs or DEFAULT_PDFS)]
    output_path = Path(args.output)

    if args.max_tokens < 200:
        raise SystemExit("--max-tokens should be at least 200 for this corpus.")
    if args.overlap_tokens < 0 or args.overlap_tokens >= args.max_tokens // 2:
        raise SystemExit("--overlap-tokens must be >= 0 and less than half of --max-tokens.")

    all_chunks: list[Chunk] = []
    all_stats: list[dict] = []
    md_dir = Path(args.markdown_dir) if args.emit_markdown else None

    for pdf_path in input_paths:
        print(f"[chunker] Processing: {pdf_path}", file=sys.stderr)
        chunks, stats = process_pdf(
            path=pdf_path,
            max_tokens=args.max_tokens,
            overlap_tokens=args.overlap_tokens,
            min_tokens=args.min_tokens,
            table_mode=args.tables,
            emit_markdown_dir=md_dir,
        )
        all_chunks.extend(chunks)
        all_stats.append(stats)
        print(
            f"[chunker] {stats['source_file']}: {stats['pages']} pages -> "
            f"{stats['sections_with_content']} sections -> {stats['chunks']} chunks",
            file=sys.stderr,
        )

    link_global_neighbors(all_chunks)

    if output_path.suffix.lower() == ".json":
        write_json(output_path, all_chunks)
    else:
        write_jsonl(output_path, all_chunks)
    write_manifest(output_path, all_stats, all_chunks, args)

    print(f"[chunker] Wrote {len(all_chunks)} chunks to {output_path}", file=sys.stderr)
    print(f"[chunker] Manifest: {output_path.with_suffix(output_path.suffix + '.manifest.json')}", file=sys.stderr)
    if args.emit_markdown:
        print(f"[chunker] Parsed Markdown: {md_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
