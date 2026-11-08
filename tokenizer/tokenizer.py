class CharacterTokenizer:

    def __init__(self, text):
        # Create a sorted vocabulary of unique characters
        self.vocab = sorted(set(text))

        # Map each character to a unique integer
        self.char_to_id = {
            char: idx
            for idx, char in enumerate(self.vocab)
        }

        # Map each integer back to its character
        self.id_to_char = {
            idx: char
            for char, idx in self.char_to_id.items()
        }

        # Total number of unique characters
        self.vocab_size = len(self.vocab)

    def encode(self, text):
        """Convert text into a list of token IDs."""
        return [self.char_to_id[char] for char in text]

    def decode(self, token_ids):
        """Convert token IDs back into text."""
        return "".join(
            self.id_to_char[token_id]
            for token_id in token_ids
        )