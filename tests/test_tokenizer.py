from tokenizer.tokenizer import CharacterTokenizer


def test_vocabulary_size():
    tokenizer = CharacterTokenizer("hello")

    assert tokenizer.vocab_size == 4


def test_encode():
    tokenizer = CharacterTokenizer("hello")

    encoded = tokenizer.encode("hello")

    assert encoded == [1, 0, 2, 2, 3]


def test_decode():
    tokenizer = CharacterTokenizer("hello")

    encoded = [1, 0, 2, 2, 3]

    decoded = tokenizer.decode(encoded)

    assert decoded == "hello"


def test_encode_decode_roundtrip():
    text = "hello world"

    tokenizer = CharacterTokenizer(text)

    encoded = tokenizer.encode(text)

    decoded = tokenizer.decode(encoded)

    assert decoded == text