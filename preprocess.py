import unicodedata
import re


def _unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')


def transform(s):
    # Lower case, trim, turn unicode to ASCII
    s = _unicode_to_ascii(s.lower().strip())
    # Add space for punctuations
    s = re.sub(r"([,.!?])", r" \1 ", s)
    # Remove non letter word
    s = re.sub(r"[^a-zA-Z,.!?]+", r" ", s)
    # Remove extra space
    s = re.sub(r"\s+", r" ", s).strip()
    return s
