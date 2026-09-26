from src.text import tokenize


def test_tokenize_keeps_prefixed_section_id_whole():
    assert tokenize("R302.1 Exterior walls.") == ["r302.1", "exterior", "walls"]


def test_tokenize_keeps_dotted_section_id_whole():
    assert tokenize("404.2.3 Clear Width.") == ["404.2.3", "clear", "width"]


def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("What does R302.1 require?") == ["what", "does", "r302.1", "require"]


def test_tokenize_splits_regulation_paragraph_letters():
    assert tokenize("Section 36.406(f)") == ["section", "36.406", "f"]
