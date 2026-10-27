import pytest

from src.ingestion.chunking import chunk_spans, chunk_units
from src.models import TextUnit

TEXT = " ".join(f"Sentence number {i} talks about topic {i}." for i in range(1, 41))


def test_chunks_respect_size_limit():
    for start, end in chunk_spans(TEXT, 200, 40):
        assert end - start <= 200


def test_chunks_cover_all_text():
    spans = chunk_spans(TEXT, 200, 0)
    covered = " ".join(TEXT[s:e] for s, e in spans)
    for i in range(1, 41):
        assert f"Sentence number {i} " in covered


def test_zero_overlap_produces_disjoint_chunks():
    spans = chunk_spans(TEXT, 200, 0)
    for (s1, e1), (s2, _) in zip(spans, spans[1:]):
        assert s2 >= e1


def test_overlap_repeats_trailing_sentences():
    spans = chunk_spans(TEXT, 200, 80)
    overlaps = [e1 - s2 for (_, e1), (s2, _) in zip(spans, spans[1:])]
    assert all(o > 0 for o in overlaps)
    assert all(o <= 80 for o in overlaps)


def test_more_overlap_means_more_chunks():
    assert len(chunk_spans(TEXT, 200, 80)) > len(chunk_spans(TEXT, 200, 0))


def test_overlong_sentence_is_hard_split_at_whitespace():
    long_sentence = "word " * 200
    spans = chunk_spans(long_sentence.strip(), 100, 20)
    assert len(spans) > 1
    assert all(e - s <= 100 for s, e in spans)
    assert all(not long_sentence[s:e].startswith(" ") for s, e in spans)


def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        chunk_spans(TEXT, 100, 100)
    with pytest.raises(ValueError):
        chunk_spans(TEXT, 0, 0)


def test_chunks_never_cross_pages_and_keep_metadata():
    units = [TextUnit("Page one text. " * 5, page_number=1), TextUnit("Page two text. " * 5, page_number=2, section="Intro")]
    chunks = chunk_units("doc123", units, 60, 10)
    assert {c.page_number for c in chunks} == {1, 2}
    for c in chunks:
        assert ("one" in c.text) == (c.page_number == 1)
        if c.page_number == 2:
            assert c.section == "Intro"
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert all(c.chunk_id.startswith("doc123-cs60_ov10-") for c in chunks)


def test_chunk_offsets_point_into_unit_text():
    unit = TextUnit(TEXT, page_number=3)
    for c in chunk_units("d", [unit], 150, 30):
        assert unit.text[c.char_start : c.char_end] == c.text
