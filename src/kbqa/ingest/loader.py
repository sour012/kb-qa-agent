"""语料加载：markdown / txt / docx / pdf 四种，统一成纯文本 + 标题。

docx 和 pdf 故意不引第三方库：
- docx 就是 zip 里的 word/document.xml，抓 <w:t> 即可，够用且零依赖；
- pdf 用已装好的 pypdfium2 取文本层。
这样别人 clone 下来只装 jieba + scikit-learn 就能跑，不会卡在编译环境上。
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

from ..models import Document

SUPPORTED = {".md", ".markdown", ".txt", ".docx", ".pdf"}


def _read_txt(path: Path) -> str:
    for enc in ("utf-8", "gb18030", "utf-16"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_docx(path: Path) -> str:
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8", errors="ignore")
    # 段落边界换成换行，run 内文本直接拼接
    paras = re.findall(r"<w:p [^>]*>[\s\S]*?</w:p>|<w:p/>", xml)
    lines: list[str] = []
    for p in paras:
        text = "".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", p))
        text = _unescape(text).strip()
        if text:
            lines.append(text)
    return "\n".join(lines)


def _unescape(s: str) -> str:
    return (
        s.replace("&lt;", "<").replace("&gt;", ">")
        .replace("&amp;", "&").replace("&quot;", '"').replace("&apos;", "'")
    )


def _read_pdf(path: Path) -> str:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    out: list[str] = []
    for i in range(len(pdf)):
        page = pdf[i]
        tp = page.get_textpage()
        out.append(tp.get_text_range())
        tp.close()
        page.close()
    pdf.close()
    return "\n".join(out)


def load_document(path: Path, corpus_dir: Path) -> Document:
    ext = path.suffix.lower()
    if ext in {".md", ".markdown", ".txt"}:
        text = _read_txt(path)
    elif ext == ".docx":
        text = _read_docx(path)
    elif ext == ".pdf":
        text = _read_pdf(path)
    else:
        raise ValueError(f"不支持的格式: {path.suffix}")

    text = text.replace("\r\n", "\n").replace("\xa0", " ")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    doc_id = path.stem
    title = _guess_title(text, doc_id)
    try:
        rel = str(path.relative_to(corpus_dir))
    except ValueError:
        rel = path.name
    return Document(doc_id=doc_id, path=rel, title=title, text=text, mtime=path.stat().st_mtime)


def _guess_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("# "):
            return s[2:].strip()
        if s and len(s) <= 40 and not s.startswith("#"):
            return s
    return fallback


def load_corpus(corpus_dir: Path) -> list[Document]:
    if not corpus_dir.exists():
        raise FileNotFoundError(f"语料目录不存在: {corpus_dir}")
    files = sorted(
        p for p in corpus_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED
    )
    docs = [load_document(p, corpus_dir) for p in files]
    return [d for d in docs if d.text.strip()]
