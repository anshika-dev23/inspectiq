from src.text import tokenize


def test_tokenize_keeps_prefixed_section_id_whole():
    assert tokenize("R302.1 Exterior walls.") == ["r302.1", "exterior", "walls"]


def test_tokenize_keeps_dotted_section_id_whole():
    assert tokenize("404.2.3 Clear Width.") == ["404.2.3", "clear", "width"]


def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("What does R302.1 require?") == ["what", "does", "r302.1", "require"]


def test_tokenize_splits_regulation_paragraph_letters():
    assert tokenize("Section 36.406(f)") == ["section", "36.406", "f"]


def test_tokenize_keeps_stopwords_by_default():
    assert tokenize("What is the width") == ["what", "is", "the", "width"]


def test_tokenize_removes_stopwords_for_bm25():
    assert tokenize("What is the minimum clear width of a door?", remove_stopwords=True) == [
        "minimum", "clear", "width", "a", "door",
    ]


def test_stopword_removal_never_drops_numbers_ids_or_normative_words():
    tokens = tokenize("What does 35.151(a) say? It shall not be 32 inches; see R302.1 and 404.2.3.",
                      remove_stopwords=True)
    assert tokens == ["35.151", "a", "say", "shall", "not", "32", "inches", "see", "r302.1", "404.2.3"]
