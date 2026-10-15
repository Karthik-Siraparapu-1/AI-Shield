import html
import re

_URL = re.compile(r"(https?://\S+|www\.\S+)", re.I)
_TAG = re.compile(r"<[^>]+>")
_NUM = re.compile(r"\d{5,}")
_WS = re.compile(r"\s+")


def clean_text(text) -> str:
    """Normalise a message: unescape HTML, strip tags, mask URLs/long numbers, lowercase."""
    if text is None or (isinstance(text, float) and text != text):
        return ""
    t = html.unescape(str(text))
    t = _TAG.sub(" ", t)
    t = _URL.sub(" urltoken ", t)
    t = _NUM.sub(" longnumber ", t)
    t = t.lower().replace("’", "'")
    return _WS.sub(" ", t).strip()
