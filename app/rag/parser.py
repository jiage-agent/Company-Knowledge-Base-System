"""
文档解析层：多格式 -> 统一 DocumentBlock。

为什么要"统一 DocumentBlock"（面试可讲）：
    每种格式的解析方式完全不同（PDF 有页、Excel 有表、Word 有标题层级）。
    如果每种的解析结果直接喂给切分器，切分器就得写四套分支逻辑。
    抽出一个统一中间结构后，下游（切分、向量化、入库）只认一种格式，
    新增格式（PPT/HTML）只需要再加一个 Parser，不动下游任何代码 —— 这就是解析器工厂的价值。

DocumentBlock 字段设计：
    block_type: text | title | table   —— 表格不能按句号切，必须单独处理
    page_no:    PDF 页码，用于"引用精确到页"
    section:    所属章节标题，用于理解上下文与展示
    source:     原始文件名，用于引用展示
"""

from __future__ import annotations

import csv
import io
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DocumentBlock:
    """解析后的统一块结构。"""

    text: str
    block_type: str = "text"          # text | title | table
    page_no: int | None = None
    section: str | None = None
    source: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.text or not self.text.strip()


class BaseParser(ABC):
    """所有格式解析器的接口。"""

    name = "base"
    suffixes: tuple[str, ...] = ()

    @abstractmethod
    def parse(self, path: Path) -> list[DocumentBlock]:
        ...

    def supports(self, suffix: str) -> bool:
        return suffix.lower() in self.suffixes


# ============================================================
# PDF
# ============================================================
class PdfParser(BaseParser):
    """
    PDF：按页提取文本。

    工程权衡：这里用 pypdf 做纯文本抽取（轻量、无系统依赖）。
    真实场景还要处理：
      - 扫描件/图片型 PDF -> 需要 OCR（pytesseract + tesseract 二进制）
      - 复杂排版表格      -> 需要版面分析（pdfplumber 的 extract_tables / 商业 OCR）
    所以这里做了一个"文本量过少就判定为疑似扫描件"的兜底提示，
    而不是假装解析成功 —— 静默产出空块比报错更可怕。
    """

    name = "pdf"
    suffixes = (".pdf",)

    MIN_CHARS_PER_PAGE = 10   # 平均每页少于此字符数，判定为疑似扫描件

    def parse(self, path: Path) -> list[DocumentBlock]:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        blocks: list[DocumentBlock] = []
        total_chars = 0

        for page_idx, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as e:  # noqa: BLE001
                logger.warning(f"第 {page_idx} 页解析失败: {e}")
                continue
            text = _clean_text(text)
            total_chars += len(text)
            if not text:
                continue
            blocks.append(
                DocumentBlock(
                    text=text,
                    block_type="text",
                    page_no=page_idx,
                    section=f"第 {page_idx} 页",
                    source=path.name,
                )
            )

        page_count = len(reader.pages) or 1
        if total_chars / page_count < self.MIN_CHARS_PER_PAGE:
            raise ValueError(
                f"解析出的文本过少（{total_chars} 字符 / {page_count} 页），"
                f"该 PDF 可能是扫描件或图片型 PDF，需要 OCR 支持（当前版本未启用）"
            )
        logger.info(f"PDF 解析完成: {path.name} -> {len(blocks)} 块 / {total_chars} 字符")
        return blocks


# ============================================================
# Word
# ============================================================
class DocxParser(BaseParser):
    """
    Word：段落 + 表格。

    关键点：按 Word 的样式名识别标题（Heading 1/2/3 -> 章节），
    这样切分时能"按章节切"而不是"按字数硬切"，
    并且每块都能带上所属章节，检索命中后展示更清楚。
    """

    name = "docx"
    suffixes = (".docx",)

    def parse(self, path: Path) -> list[DocumentBlock]:
        from docx import Document as DocxDocument
        from docx.table import Table
        from docx.text.paragraph import Paragraph
        from docx.oxml.ns import qn

        doc = DocxDocument(str(path))
        blocks: list[DocumentBlock] = []
        current_section: str | None = None
        page_guess = 1

        def iter_children(parent):
            for child in parent.element.body.iterchildren():
                if child.tag == qn("w:p"):
                    yield Paragraph(child, parent)
                elif child.tag == qn("w:tbl"):
                    yield Table(child, parent)

        for item in iter_children(doc):
            if isinstance(item, Paragraph):
                text = _clean_text(item.text)
                if not text:
                    continue
                style = (item.style.name or "").lower()
                is_heading = style.startswith("heading") or style.startswith("标题")
                if is_heading:
                    current_section = text
                    blocks.append(
                        DocumentBlock(text=text, block_type="title",
                                      page_no=page_guess, section=text, source=path.name)
                    )
                else:
                    blocks.append(
                        DocumentBlock(text=text, block_type="text",
                                      page_no=page_guess, section=current_section, source=path.name)
                    )
            else:  # 表格
                rows = []
                for row in item.rows:
                    cells = [_clean_text(c.text) for c in row.cells]
                    if any(cells):
                        rows.append(" | ".join(cells))
                if rows:
                    # 表格整体作为一个块：不能被句号切碎，否则表头与数据行会分离
                    blocks.append(
                        DocumentBlock(
                            text="\n".join(rows),
                            block_type="table",
                            page_no=page_guess,
                            section=current_section,
                            source=path.name,
                            meta={"rows": len(rows)},
                        )
                    )

        logger.info(f"DOCX 解析完成: {path.name} -> {len(blocks)} 块")
        return blocks


# ============================================================
# Excel
# ============================================================
class ExcelParser(BaseParser):
    """
    Excel：按工作表 + 按行输出。

    为什么按"行"而不是按"单元格"：
      一行代表一条完整记录（如"职级: P5 | 年假: 10天"），
      语义自包含，检索命中后答案完整；拆到单元格就丢语义了。
    同时保留表头，让每行自带列名（"年假: 10天"而不是孤零零的 "10天"）。
    """

    name = "xlsx"
    suffixes = (".xlsx", ".xlsm")

    def parse(self, path: Path) -> list[DocumentBlock]:
        from openpyxl import load_workbook

        wb = load_workbook(str(path), read_only=True, data_only=True)
        blocks: list[DocumentBlock] = []

        for ws in wb.worksheets:
            rows = list(ws.iter_rows(values_only=True))
            if not rows:
                continue
            header = [str(c).strip() if c is not None else f"列{i+1}" for i, c in enumerate(rows[0])]
            blocks.append(
                DocumentBlock(
                    text=f"工作表：{ws.title}\n表头：" + " | ".join(header),
                    block_type="title",
                    page_no=1,
                    section=ws.title,
                    source=path.name,
                )
            )
            for row_idx, row in enumerate(rows[1:], start=2):
                pairs = []
                for col_idx, cell in enumerate(row):
                    if cell is None or str(cell).strip() == "":
                        continue
                    col_name = header[col_idx] if col_idx < len(header) else f"列{col_idx+1}"
                    pairs.append(f"{col_name}：{str(cell).strip()}")
                if not pairs:
                    continue
                blocks.append(
                    DocumentBlock(
                        text=f"[{ws.title} 第{row_idx}行] " + "；".join(pairs),
                        block_type="table",
                        page_no=1,
                        section=ws.title,
                        source=path.name,
                        meta={"row": row_idx, "sheet": ws.title},
                    )
                )
        wb.close()
        logger.info(f"Excel 解析完成: {path.name} -> {len(blocks)} 块")
        return blocks


# ============================================================
# 纯文本 / Markdown
# ============================================================
class TextParser(BaseParser):
    """
    文本/Markdown：按空行分段，并识别章节标题（第X章 / 第X条 / 一、 / Markdown #）。

    加 Markdown 支持是有意的：写项目文档、爬取内部 Wiki 导出都是 md 格式，
    演示时"上传自己写的 README 让它回答"效果很好。
    """

    name = "text"
    suffixes = (".txt", ".md", ".markdown", ".log")

    def parse(self, path: Path) -> list[DocumentBlock]:
        for enc in ("utf-8", "gbk", "utf-16"):
            try:
                raw = path.read_text(encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            raw = path.read_text(encoding="utf-8", errors="ignore")

        blocks: list[DocumentBlock] = []
        current_section: str | None = None

        for para in raw.split("\n\n"):
            text = _clean_text(para)
            if not text:
                continue

            # 段落内部：先看有没有章节标题行，有就按标题切成"标题块 + 正文块"。
            # 这样切分器能拿到明确的章节边界（section），
            # 每个子块也能带上"我属于哪一章"的元数据，引用展示更清楚。
            segments: list[tuple[str, str]] = []      # (类型, 文本)
            buf: list[str] = []
            for line in text.split("\n"):
                if _is_heading(line):
                    if buf:
                        segments.append(("text", "\n".join(buf)))
                        buf = []
                    segments.append(("title", line.strip()))
                else:
                    buf.append(line)
            if buf:
                segments.append(("text", "\n".join(buf)))

            for kind, seg_text in segments:
                seg_text = seg_text.strip()
                if not seg_text:
                    continue
                if kind == "title":
                    current_section = seg_text.lstrip("# ").strip()
                    blocks.append(DocumentBlock(
                        text=current_section, block_type="title",
                        section=current_section, source=path.name))
                else:
                    blocks.append(DocumentBlock(
                        text=seg_text, block_type="text",
                        section=current_section, source=path.name))

        logger.info(f"文本解析完成: {path.name} -> {len(blocks)} 块 "
                    f"(其中标题块 {sum(1 for b in blocks if b.block_type == 'title')} 个)")
        return blocks


class CsvParser(BaseParser):
    name = "csv"
    suffixes = (".csv",)

    def parse(self, path: Path) -> list[DocumentBlock]:
        blocks: list[DocumentBlock] = []
        for enc in ("utf-8-sig", "gbk"):
            try:
                text = path.read_text(encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")

        reader = csv.reader(io.StringIO(text))
        rows = list(reader)
        if not rows:
            return blocks
        header = [h.strip() or f"列{i+1}" for i, h in enumerate(rows[0])]
        for row_idx, row in enumerate(rows[1:], start=2):
            pairs = [f"{header[i] if i < len(header) else f'列{i+1}'}：{c.strip()}"
                     for i, c in enumerate(row) if c.strip()]
            if pairs:
                blocks.append(
                    DocumentBlock(text=f"[第{row_idx}行] " + "；".join(pairs),
                                  block_type="table", section=path.stem, source=path.name)
                )
        return blocks


# ============================================================
# 工厂
# ============================================================
_PARSERS: list[BaseParser] = [
    PdfParser(),
    DocxParser(),
    ExcelParser(),
    TextParser(),
    CsvParser(),
]

SUPPORTED_SUFFIXES = tuple(s for p in _PARSERS for s in p.suffixes)


def get_parser(filename: str) -> BaseParser:
    """
    解析器路由（parser factory）：按扩展名选解析器。

    面试延伸问题：「格式识别怎么做的？」
    答：先按扩展名路由，再在解析后做内容校验（PDF 会检查文本量、Excel 会检查是否有表）；
    更严格的场景会用 python-magic 读文件头（magic number）防止"改名骗过系统"，
    这里用扩展名 + 内容兜底校验，是成本与严谨度的折中。
    """
    suffix = Path(filename).suffix.lower()
    for p in _PARSERS:
        if p.supports(suffix):
            return p
    raise ValueError(f"不支持的格式: {suffix}，当前支持: {', '.join(SUPPORTED_SUFFIXES)}")


def parse_document(path: Path) -> list[DocumentBlock]:
    parser = get_parser(path.name)
    blocks = parser.parse(path)
    if not blocks:
        raise ValueError(f"文档解析后没有任何内容: {path.name}")
    return blocks


# 章节标题识别（中文文档的层级结构标记）。
# 为什么必须做这一步（实测踩到的坑）：
#   "第一章 总则" 在纯文本里往往混在行中间/行首，文本解析器并不知道它是结构标记；
#   如果只按空行分段，整篇文档会变成一个大块，切分时无法按章节断开，
#   导致"住宿标准"这类条款被裹在 900 字巨块里、被其他主题稀释，检索命中率下降。
#
# ⚠️ 只识别【章/节/篇/部分/一、/Markdown #】这类结构标题，
#    绝不把 "第X条" 当标题（实测踩坑）：
#    "第一条 员工入职满一年后享受带薪年休假。" 是**正文条款**而不是标题，
#    如果把它判成 title，这块正文就会被丢弃（title 块不参与向量化），
#    结果是"文档看着切了很多块，但关键条款全丢了"——这是最隐蔽也最致命的一类 bug。
_HEADING_PATTERNS = [
    # "第一章 总则" / "第 2 节 概述" / "第三部分 附录"（要求标题短，避免误吞正文）
    re.compile(r"^第\s*[一二三四五六七八九十百零\d]+\s*[章节篇]\s*[、\.．:：]?\s*\S{0,20}$"),
    re.compile(r"^第\s*[一二三四五六七八九十百零\d]+\s*部分\s*[、\.．:：]?\s*\S{0,20}$"),
    # "一、总则" / "（二）适用范围"
    re.compile(r"^[一二三四五六七八九十]+\s*[、\.．]\s*\S{0,20}$"),
    re.compile(r"^[（(]\s*[一二三四五六七八九十\d]+\s*[)）]\s*\S{0,20}$"),
    # Markdown 标题
    re.compile(r"^#{1,6}\s+\S+$"),
]


def _is_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 30:      # 标题不可能很长，长了就是正文
        return False
    if line.endswith(("。", "；", "！", "？", ".", ";")):   # 以句号结尾的是正文
        return False
    return any(p.match(line) for p in _HEADING_PATTERNS)


def _clean_text(text: str) -> str:
    """
    文本清洗。

    为什么必须做：PDF 复制出来的文本常带大量空白、软换行、
    以及页眉页脚噪声。脏文本会直接拉低 embedding 质量
    （噪声稀释语义向量），也会浪费 token。
    """
    if not text:
        return ""
    # 全角空格 -> 半角；去掉零宽字符与 BOM
    text = text.replace("\u3000", " ").replace("\ufeff", "").replace("\u200b", "")
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            lines.append(stripped)
    result = "\n".join(lines)
    # 压缩连续空格
    while "  " in result:
        result = result.replace("  ", " ")
    return result.strip()
