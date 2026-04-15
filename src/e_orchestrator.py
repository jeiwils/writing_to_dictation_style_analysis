"""Dictation-focused end-to-end pipeline.

This module is a filtered copy of the general orchestrator logic containing only
the stages required for writing-to-dictation analysis:
preprocessing, corpus metrics, window metrics, and dictation visualisations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from spacy.tokens import Doc
from tqdm import tqdm

from .a_preprocessing_cleaning import preprocess_all_texts
from .b_log_prob_metrics import WholeTextMetrics
from .c1_syntactics import SyntaxAnalyzer
from .c2_lexico_semantics import LexicoSemanticsAnalyzer
from .c3_discourse import DiscourseAnalyzer
from .x_configs import DEFAULT_MATTR_WINDOW_SIZE, DEFAULT_USE_EXISTING, DEFAULT_WINDOW_SIZE
from .z_utils import analytics_path, iter_dirs, load_spacy_model, text_path, window_metrics_filename

from .f_text_dictation_visualisations import generate_all_dictation_visualisations


DICTATION_STAGE_SWITCHES = {
    "preprocessing": True,
    "corpus_metrics": True,
    "window_metrics": True,
    "dictation_visualisations": True,
}


WINDOW_ANALYSIS_SWITCHES = {
    "syntax": True,
    "lexico_semantics": True,
    "discourse": True,
    "log_prob": True,
}


def _merge_switches(defaults: Dict[str, bool], overrides: Optional[Dict[str, bool]] = None) -> Dict[str, bool]:
    merged = dict(defaults)
    if overrides:
        for key, value in overrides.items():
            if key in merged:
                merged[key] = bool(value)
    return merged


def run_preprocessing(process_unknown=True, use_existing=DEFAULT_USE_EXISTING, authors=None):
    """Run TXT raw-text preprocessing to produce cleaned/normalised corpora."""
    preprocess_all_texts(
        process_unknown=process_unknown,
        use_existing=use_existing,
        authors=authors,
    )


def _load_segmented_jsonl(path: Path) -> List[str]:
    sentences: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = entry.get("text") if isinstance(entry, dict) else None
            if text:
                sentences.append(str(text).strip())
    return [s for s in sentences if s]


def run_corpus_metrics(use_existing=DEFAULT_USE_EXISTING, authors=None):
    """Compute and save corpus-level log-prob/surprisal metrics for cleaned texts."""
    window_size = DEFAULT_WINDOW_SIZE
    metrics = WholeTextMetrics()
    nlp = load_spacy_model()

    cleaned_root = text_path("processed", "cleaned_texts")
    segmented_root = text_path("processed", "cleaned_segmented_texts")
    output_root = analytics_path("corpus")
    output_root.mkdir(parents=True, exist_ok=True)

    categories = list(iter_dirs(cleaned_root, authors=authors, depth=2))
    processed = 0
    skipped = 0
    for category_key, subdir in tqdm(categories, desc="Corpus metrics", ascii=True):
        genre, author = category_key.split("/", 1)

        out_subdir = output_root / genre / author
        out_subdir.mkdir(parents=True, exist_ok=True)

        cleaned_files = sorted(subdir.glob("*.json"))

        file_texts = []
        for file in cleaned_files:
            try:
                text = json.load(file.open("r", encoding="utf-8")).get("text", "")
            except json.JSONDecodeError:
                text = ""
            file_texts.append((file, text))

        for file, text in tqdm(
            file_texts,
            desc=f"Corpus metrics: {category_key}",
            leave=False,
            ascii=True,
        ):
            base_name = file.stem.replace("_cleaned", "")
            text_dir = out_subdir / base_name
            text_dir.mkdir(parents=True, exist_ok=True)

            freq_path = text_dir / "corpus_frequencies.json"
            if use_existing and freq_path.exists():
                try:
                    freqs = json.load(freq_path.open("r", encoding="utf-8"))
                except json.JSONDecodeError:
                    freqs = {}
            else:
                freqs = metrics.compute_corpus_frequencies([text], nlp=nlp)
                freq_path.parent.mkdir(parents=True, exist_ok=True)
                freq_path.write_text(json.dumps(freqs, indent=2), encoding="utf-8")

            output_file = text_dir / "corpus_metrics.json"
            if use_existing and output_file.exists():
                skipped += 1
                continue

            segmented_path = segmented_root / genre / author / f"{base_name}_cleaned_segmented.jsonl"
            if segmented_path.exists():
                segmented_sentences = _load_segmented_jsonl(segmented_path)
                text = "\n".join(segmented_sentences)
                spans = []
                cursor = 0
                for sent in segmented_sentences:
                    end = cursor + len(sent)
                    spans.append((cursor, end))
                    cursor = end + 1
            else:
                spans = None

            result = metrics.build_metrics_for_text(
                text,
                file.name,
                nlp=nlp,
                window_size=window_size,
                sentence_spans=spans,
            )
            output_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
            processed += 1
    tqdm.write(f"Corpus metrics complete: {processed} processed, {skipped} skipped.")


def _resolve_corpus_base_name(metrics_file: Path) -> str:
    if metrics_file.name != "corpus_metrics.json":
        raise ValueError(f"Expected corpus_metrics.json; got {metrics_file.name}")
    return metrics_file.parent.name


def _load_text_for_windowing(category_key: str, metrics_file: Path) -> Tuple[str, List[str]]:
    segmented_root = text_path("processed", "cleaned_segmented_texts", category_key)
    base_name = _resolve_corpus_base_name(metrics_file)
    candidate = segmented_root / f"{base_name}_cleaned_segmented.jsonl"
    if candidate.exists():
        sentences = _load_segmented_jsonl(candidate)
        return "\n".join(sentences), sentences
    raise FileNotFoundError(
        f"Segmented text not found for {metrics_file.name}: expected {candidate}. "
        "Generate cleaned segmented texts before running window metrics."
    )


def _load_corpus_frequencies(text_dir: Path) -> dict:
    freq_path = text_dir / "corpus_frequencies.json"
    if not freq_path.exists():
        return {}
    try:
        data = json.load(freq_path.open("r", encoding="utf-8"))
        if isinstance(data, dict) and "word_frequencies" in data:
            return data.get("word_frequencies") or {}
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def run_windowed_metrics(
    mattr_window_size=DEFAULT_MATTR_WINDOW_SIZE,
    use_existing=DEFAULT_USE_EXISTING,
    authors=None,
    analysis_sections=None,
):
    """Compute and save dictation-relevant window metrics from corpus outputs."""
    analysis_sections = _merge_switches(WINDOW_ANALYSIS_SWITCHES, analysis_sections)
    enabled_domains = [name for name, enabled in analysis_sections.items() if enabled]
    if not enabled_domains:
        tqdm.write("Window metrics skipped: no analysis sections enabled.")
        return

    window_size = DEFAULT_WINDOW_SIZE
    corpus_root = analytics_path("corpus")
    output_root = analytics_path("window")
    output_root.mkdir(parents=True, exist_ok=True)

    nlp = load_spacy_model()
    syntax_analyzer = SyntaxAnalyzer(nlp) if analysis_sections.get("syntax") else None
    discourse_analyzer = DiscourseAnalyzer(nlp) if analysis_sections.get("discourse") else None

    categories = list(iter_dirs(corpus_root, authors=authors, depth=2))
    processed = 0
    skipped = 0
    for category_key, author_dir in tqdm(categories, desc="Window metrics", ascii=True):
        genre, author = category_key.split("/", 1)
        out_category_dir = output_root / genre / author
        out_category_dir.mkdir(parents=True, exist_ok=True)

        metric_files = []
        for text_dir in author_dir.iterdir():
            if not text_dir.is_dir():
                continue
            metric_files.extend(sorted(text_dir.glob("corpus_metrics.json")))
        for file in tqdm(
            metric_files,
            desc=f"Window metrics: {category_key}",
            leave=False,
            ascii=True,
        ):
            text_dir = file.parent
            base_name = _resolve_corpus_base_name(file)
            corpus_freqs = _load_corpus_frequencies(text_dir)
            global_avg_freq = (sum(corpus_freqs.values()) / len(corpus_freqs)) if corpus_freqs else None
            lex_analyzer = (
                LexicoSemanticsAnalyzer(nlp, corpus_freqs=corpus_freqs)
                if analysis_sections.get("lexico_semantics")
                else None
            )

            output_text_dir = out_category_dir / base_name
            output_text_dir.mkdir(parents=True, exist_ok=True)
            output_files = {domain: output_text_dir / window_metrics_filename(domain) for domain in enabled_domains}
            if use_existing and output_files and all(path.exists() for path in output_files.values()):
                skipped += 1
                continue

            data = json.load(file.open("r", encoding="utf-8"))
            meta_block = data.get("meta", {}) if isinstance(data, dict) else {}
            _, segmented_sentences = _load_text_for_windowing(category_key, file)
            if not segmented_sentences:
                raise ValueError(
                    f"No segmented sentences found for {file.name}; expected cleaned segmented JSONL."
                )

            tokenized_docs = [nlp.make_doc(text) for text in segmented_sentences]
            for sent_doc in tokenized_docs:
                for i, token in enumerate(sent_doc):
                    token.is_sent_start = i == 0

            doc = Doc.from_docs(tokenized_docs)
            for name, proc in nlp.pipeline:
                if name == "senter":
                    continue
                doc = proc(doc)
            num_sentences = len(segmented_sentences)
            doc_sentence_count = len(list(doc.sents))
            if doc_sentence_count != num_sentences:
                raise ValueError(
                    f"Sentence count mismatch after spaCy pipeline for {file.name}: "
                    f"segmented={num_sentences}, parsed={doc_sentence_count}. "
                    "Check sentence boundary drift or adjust pipeline components."
                )

            syntax_metrics = (
                syntax_analyzer.analyze_document(doc, window_size=window_size)
                if syntax_analyzer is not None
                else None
            )
            lex_metrics = (
                lex_analyzer.analyze_document(
                    doc,
                    window_size=window_size,
                    mattr_window_size=mattr_window_size,
                    global_avg_freq=global_avg_freq,
                )
                if lex_analyzer is not None
                else None
            )
            if discourse_analyzer is not None:
                discourse_sent, discourse_windows = discourse_analyzer.compute_sentence_metrics(doc, window_size=window_size)
                discourse_metrics = {
                    "meta": {"window_size": window_size, "num_sentences": num_sentences},
                    "sentences": discourse_sent,
                    "windows": discourse_windows,
                }
            else:
                discourse_metrics = None

            log_prob_sentences = data.get("sentences", [])
            log_prob_windows = data.get("windows", [])
            base_meta = {
                "filename": meta_block.get("filename", file.name),
                "model": meta_block.get("model", ""),
                "window_size": window_size,
                "num_sentences": num_sentences,
            }
            log_prob_meta = {
                **base_meta,
                "num_sentences": len(log_prob_sentences) if log_prob_sentences else num_sentences,
                "avg_log_prob": meta_block.get("avg_log_prob"),
            }

            def _domain_payload(metrics: Dict[str, object], windows: List[Dict[str, object]]) -> Dict[str, object]:
                meta = metrics.get("meta") if isinstance(metrics, dict) else {}
                merged_meta = {**base_meta, **(meta if isinstance(meta, dict) else {})}
                return {
                    "meta": merged_meta,
                    "sentences": metrics.get("sentences", []),
                    "windows": windows,
                }

            if syntax_metrics is not None:
                syntax_payload = _domain_payload(syntax_metrics, syntax_metrics.get("windows", []))
                with open(output_files["syntax"], "w", encoding="utf-8") as f:
                    json.dump(syntax_payload, f, indent=2)

            if lex_metrics is not None:
                lexico_payload = _domain_payload(lex_metrics, lex_metrics.get("windows", []))
                with open(output_files["lexico_semantics"], "w", encoding="utf-8") as f:
                    json.dump(lexico_payload, f, indent=2)

            if discourse_metrics is not None:
                discourse_payload = _domain_payload(discourse_metrics, discourse_metrics.get("windows", []))
                with open(output_files["discourse"], "w", encoding="utf-8") as f:
                    json.dump(discourse_payload, f, indent=2)

            if analysis_sections.get("log_prob"):
                log_prob_payload = {
                    "meta": log_prob_meta,
                    "sentences": log_prob_sentences,
                    "windows": log_prob_windows,
                }
                with open(output_files["log_prob"], "w", encoding="utf-8") as f:
                    json.dump(log_prob_payload, f, indent=2)

            processed += 1
    tqdm.write(
        f"Window metrics complete: {processed} processed, {skipped} skipped. "
        f"Enabled sections: {', '.join(enabled_domains)}"
    )


def run_dictation_pipeline(
    *,
    use_existing: bool = DEFAULT_USE_EXISTING,
    process_unknown: bool = True,
    authors=None,
    mattr_window_size: int = DEFAULT_MATTR_WINDOW_SIZE,
    output_root: Optional[Path] = None,
    stage_switches: Optional[Dict[str, bool]] = None,
):
    """Run the full writing-to-dictation pipeline without dashboard stages."""
    switches = _merge_switches(DICTATION_STAGE_SWITCHES, stage_switches)

    if switches.get("preprocessing"):
        tqdm.write("Stage 1/4: preprocessing")
        run_preprocessing(
            process_unknown=process_unknown,
            use_existing=use_existing,
            authors=authors,
        )
    else:
        tqdm.write("Stage 1/4: preprocessing [skipped]")

    if switches.get("corpus_metrics"):
        tqdm.write("Stage 2/4: corpus metrics")
        run_corpus_metrics(use_existing=use_existing, authors=authors)
    else:
        tqdm.write("Stage 2/4: corpus metrics [skipped]")

    if switches.get("window_metrics"):
        tqdm.write("Stage 3/4: window metrics")
        run_windowed_metrics(
            mattr_window_size=mattr_window_size,
            use_existing=use_existing,
            authors=authors,
            analysis_sections={
                "syntax": True,
                "lexico_semantics": True,
                "discourse": True,
                "log_prob": True,
            },
        )
    else:
        tqdm.write("Stage 3/4: window metrics [skipped]")

    if switches.get("dictation_visualisations"):
        tqdm.write("Stage 4/4: dictation visualisations")
        outputs = generate_all_dictation_visualisations(
            use_existing=use_existing,
            output_root=output_root,
        )
    else:
        tqdm.write("Stage 4/4: dictation visualisations [skipped]")
        outputs = {}

    tqdm.write("Dictation pipeline complete.")
    return outputs


if __name__ == "__main__":
    run_dictation_pipeline()

