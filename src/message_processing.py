
import re
import unicodedata


TURKISH_WORDS = {
    "merhaba", "benim", "için", "nasıl",
    "neden", "garanti", "garantisi",
    "buzdolabı", "buzdolabım",
    "çamaşır", "makinesi", "bulaşık",
    "çalışmıyor", "bozuldu", "arıza",
    "tamir", "servis", "iade",
    "kargo", "teslimat", "istiyorum",
    "ediyor", "devam", "mı", "mi",
    "mu", "mü"
}

def detect_language(text):
    text = str(text).casefold()

    # Turkish-specific characters provide a useful clue.
    if re.search(r"[çğıöşü]", text):
        return "tr"

    tokens = set(
        re.findall(r"[^\W_]+", text, re.UNICODE)
    )

    matches = len(tokens & TURKISH_WORDS)

    if matches >= 1:
        return "tr"

    return "en"


def extract_product_codes(text):
    """
    Extract candidate alphanumeric product codes.

    Examples:
        WAT28400
        WW90T554DAW
        ABC-1234

    This is a heuristic, not a verified product catalog.
    """

    pattern = (
        r"\b(?=[A-Z0-9-]{5,20}\b)"
        r"(?=[A-Z0-9-]*[A-Z])"
        r"(?=[A-Z0-9-]*\d)"
        r"[A-Z0-9]+(?:-[A-Z0-9]+)*\b"
    )

    candidates = re.findall(
        pattern,
        str(text).upper()
    )

    # Remove duplicates while preserving order.
    return list(dict.fromkeys(candidates))


def normalize_message(text):
    text = unicodedata.normalize(
        "NFKC",
        str(text)
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip()

def get_small_talk_reply(message, language):
    """Return a rule-based reply for greetings and small talk."""

    import re

    text = message.casefold().strip()
    text = re.sub(r"[!?.,]+$", "", text).strip()

    greetings = {
        "hello", "hi", "hey", "good morning",
        "merhaba", "selam", "günaydın", "iyi günler",
    }

    thanks = {
        "thanks", "thank you", "thank you so much",
        "teşekkürler", "teşekkür ederim", "sağ ol", "sağol",
    }

    how_are_you = {
        "how are you", "how are you doing",
        "nasılsın", "nasılsınız",
    }

    if text in greetings:
        if language == "tr":
            return "Merhaba! Beyaz eşya ürününüzle ilgili size nasıl yardımcı olabilirim?"
        return "Hello! How can I help you with your home appliance today?"

    if text in thanks:
        if language == "tr":
            return "Rica ederim! Başka bir konuda yardımcı olabilir miyim?"
        return "You're welcome! Is there anything else I can help you with?"

    if text in how_are_you:
        if language == "tr":
            return "Teşekkür ederim! Size beyaz eşya desteği konusunda nasıl yardımcı olabilirim?"
        return "I'm here to help! What home-appliance issue can I assist you with?"

    return None

def levenshtein_distance(a, b):
    """Minimum single-character edits between two strings."""
    a = a.casefold()
    b = b.casefold()

    previous = list(range(len(b) + 1))

    for i, char_a in enumerate(a, start=1):
        current = [i]

        for j, char_b in enumerate(b, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (char_a != char_b)

            current.append(min(insertion, deletion, substitution))

        previous = current

    return previous[-1]

def correct_appliance_names(message):
    """Correct likely appliance-name typos using edit-distance matching."""
    
    import re

    appliance_names = {
        "fridge", "refrigerator", "freezer", "dishwasher",
        "washer", "dryer", "oven", "microwave",
        "buzdolabı", "bulaşık", "çamaşır", "kurutma",
        "fırın", "mikrodalga",
    }

    corrections = []

    def replace_word(match):
        original = match.group(0)
        word = original.casefold()

        if word in appliance_names:
            return original

        candidates = sorted(
            (
                (levenshtein_distance(word, name), name)
                for name in appliance_names
            ),
            key=lambda item: (item[0], item[1]),
        )

        if candidates:
            distance, corrected = candidates[0]

            max_distance = 1 if len(word) <= 6 else 2

            if distance > max_distance:
                return original

            if word[0] != corrected[0]:
                return original

            if len(word) <= 6 and word[-1] != corrected[-1]:
                return original

            if original[0].isupper():
                corrected = corrected.capitalize()

            corrections.append((original, corrected))
            return corrected

        return original

    corrected_message = re.sub(
        r"[^\W\d_]+",
        replace_word,
        message,
        flags=re.UNICODE,
    )

    return corrected_message, corrections