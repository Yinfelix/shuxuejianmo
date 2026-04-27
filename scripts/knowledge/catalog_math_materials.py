from __future__ import annotations

import json
import re
import subprocess
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

from pypdf import PdfReader


SOURCE_DIR = Path(r"d:\Downloads\数学建模资料")
KNOWLEDGE_DIR = Path("knowledge/math_materials")
CATALOG_PATH = KNOWLEDGE_DIR / "materials_catalog.json"
EXCERPTS_PATH = KNOWLEDGE_DIR / "materials_excerpts.json"
SUMMARY_PATH = KNOWLEDGE_DIR / "materials_inventory.md"
UNRAR_PATH = Path(r"C:\Program Files\WinRAR\UnRAR.exe")

ARCHIVE_SUFFIXES = {".zip", ".rar"}
PDF_SUFFIXES = {".pdf"}
WORD_SUFFIXES = {".doc", ".docx"}

EXCELLENT_KEYWORDS = [
    "优秀论文",
    "特等奖",
    "一等奖",
    "二等奖",
    "获奖论文",
    "优秀作品",
    "f奖",
    "o奖",
    "m奖",
]
MANUAL_KEYWORDS = [
    "参赛手册",
    "赛事说明",
    "报名通知",
    "规则",
    "说明",
    "指南",
    "手册",
]
PAPER_HINT_KEYWORDS = [
    "摘要",
    "关键词",
    "问题重述",
    "模型假设",
    "符号说明",
    "模型建立",
    "求解",
    "结果分析",
]


@dataclass
class FileRecord:
    path: str
    relative_path: str
    category: str
    source_archive: str | None
    extracted_dir: str | None
    is_candidate_excellent_paper: bool
    is_candidate_manual: bool
    size_bytes: int


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def classify_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in ARCHIVE_SUFFIXES:
        return "archive"
    if suffix in PDF_SUFFIXES:
        return "pdf"
    if suffix in WORD_SUFFIXES:
        return "document"
    if path.is_dir():
        return "directory"
    return "other"


def has_keywords(name: str, keywords: list[str]) -> bool:
    lowered = name.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def extract_zip(archive_path: Path, target_dir: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(target_dir)


def extract_rar(archive_path: Path, target_dir: Path) -> None:
    if not UNRAR_PATH.exists():
        raise FileNotFoundError(f"UnRAR not found: {UNRAR_PATH}")
    target_dir.mkdir(parents=True, exist_ok=True)
    command = [str(UNRAR_PATH), "x", "-o+", str(archive_path), str(target_dir)]
    subprocess.run(command, check=True, capture_output=True, text=True)


def extract_archive(archive_path: Path) -> Path:
    extracted_dir = archive_path.with_suffix("")
    if extracted_dir.exists() and any(extracted_dir.iterdir()):
        return extracted_dir
    extracted_dir.mkdir(parents=True, exist_ok=True)
    if archive_path.suffix.lower() == ".zip":
        extract_zip(archive_path, extracted_dir)
    elif archive_path.suffix.lower() == ".rar":
        extract_rar(archive_path, extracted_dir)
    else:
        raise ValueError(f"Unsupported archive type: {archive_path}")
    return extracted_dir


def read_pdf_excerpt(pdf_path: Path, max_pages: int = 3, max_chars: int = 6000) -> str:
    reader = PdfReader(str(pdf_path))
    collected: list[str] = []
    for page in reader.pages[:max_pages]:
        page_text = page.extract_text() or ""
        if page_text:
            collected.append(page_text)
    text = normalize_text(" ".join(collected))
    return text[:max_chars]


def is_excellent_paper(path: Path, excerpt: str | None) -> bool:
    if has_keywords(path.name, EXCELLENT_KEYWORDS):
        return True
    if excerpt:
        return sum(keyword in excerpt for keyword in PAPER_HINT_KEYWORDS) >= 3 and "摘要" in excerpt
    return False


def is_manual(path: Path, excerpt: str | None) -> bool:
    if has_keywords(path.name, MANUAL_KEYWORDS):
        return True
    if excerpt:
        manual_markers = ["报名", "参赛", "时间", "要求", "提交", "格式"]
        return sum(marker in excerpt for marker in manual_markers) >= 3
    return False


def build_catalog() -> tuple[list[FileRecord], list[dict[str, str | bool]]]:
    records: list[FileRecord] = []
    excerpts: list[dict[str, str | bool]] = []

    archives = sorted(path for path in SOURCE_DIR.iterdir() if path.suffix.lower() in ARCHIVE_SUFFIXES)
    extracted_map: dict[Path, Path] = {}
    for archive_path in archives:
        try:
            extracted_map[archive_path] = extract_archive(archive_path)
        except Exception as exc:  # noqa: BLE001
            excerpts.append(
                {
                    "path": str(archive_path),
                    "relative_path": str(archive_path.relative_to(SOURCE_DIR)),
                    "excerpt": f"ARCHIVE_EXTRACTION_FAILED: {exc}",
                    "is_candidate_excellent_paper": False,
                    "is_candidate_manual": False,
                }
            )

    all_paths = [SOURCE_DIR]
    all_paths.extend(sorted(extracted_map.values()))
    visited: set[Path] = set()

    for root in all_paths:
        for path in sorted(root.rglob("*")):
            if path in visited:
                continue
            visited.add(path)
            if path.is_dir():
                continue

            excerpt: str | None = None
            source_archive = None
            extracted_dir = None
            for archive_path, extraction_dir in extracted_map.items():
                if extraction_dir in path.parents:
                    source_archive = str(archive_path.relative_to(SOURCE_DIR))
                    extracted_dir = str(extraction_dir.relative_to(SOURCE_DIR))
                    break

            if path.suffix.lower() == ".pdf":
                try:
                    excerpt = read_pdf_excerpt(path)
                except Exception as exc:  # noqa: BLE001
                    excerpt = f"PDF_READ_FAILED: {exc}"

            excellent = is_excellent_paper(path, excerpt)
            manual = is_manual(path, excerpt)
            record = FileRecord(
                path=str(path),
                relative_path=str(path.relative_to(SOURCE_DIR)),
                category=classify_path(path),
                source_archive=source_archive,
                extracted_dir=extracted_dir,
                is_candidate_excellent_paper=excellent,
                is_candidate_manual=manual,
                size_bytes=path.stat().st_size,
            )
            records.append(record)
            if excerpt:
                excerpts.append(
                    {
                        "path": record.path,
                        "relative_path": record.relative_path,
                        "excerpt": excerpt,
                        "is_candidate_excellent_paper": excellent,
                        "is_candidate_manual": manual,
                    }
                )

    return records, excerpts


def write_summary(records: list[FileRecord]) -> None:
    root_files = [record for record in records if len(Path(record.relative_path).parts) == 1]
    candidate_papers = [record for record in records if record.is_candidate_excellent_paper]
    candidate_manuals = [record for record in records if record.is_candidate_manual]

    lines = [
        "# 数学建模资料目录清单",
        "",
        "## 根目录文件",
        "",
    ]
    for record in sorted(root_files, key=lambda item: item.relative_path):
        lines.append(
            f"- {record.relative_path} | {record.category} | size={record.size_bytes}"
        )

    lines.extend([
        "",
        "## 候选优秀论文",
        "",
    ])
    for record in sorted(candidate_papers, key=lambda item: item.relative_path):
        lines.append(
            f"- {record.relative_path}"
            + (f" | source_archive={record.source_archive}" if record.source_archive else "")
        )

    lines.extend([
        "",
        "## 候选参赛手册与赛事说明",
        "",
    ])
    for record in sorted(candidate_manuals, key=lambda item: item.relative_path):
        lines.append(
            f"- {record.relative_path}"
            + (f" | source_archive={record.source_archive}" if record.source_archive else "")
        )

    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    records, excerpts = build_catalog()
    CATALOG_PATH.write_text(
        json.dumps([asdict(record) for record in records], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    EXCERPTS_PATH.write_text(
        json.dumps(excerpts, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_summary(records)
    print(f"Catalog written to: {CATALOG_PATH}")
    print(f"Excerpts written to: {EXCERPTS_PATH}")
    print(f"Summary written to: {SUMMARY_PATH}")
    print(f"Total records: {len(records)}")
    print(f"Candidate excellent papers: {sum(record.is_candidate_excellent_paper for record in records)}")
    print(f"Candidate manuals: {sum(record.is_candidate_manual for record in records)}")


if __name__ == "__main__":
    main()