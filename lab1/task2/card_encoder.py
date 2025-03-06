import re
from utils.get_text_from_file import read_text_from_file


CARD_PATTERN = r'\b(\d{2})(\d{2})[\s-]?(\d{4})[\s-]?(\d{4})[\s-]?(\d{4})\b'

def mask_bank_cards(text: str) -> str:
    text = re.sub(CARD_PATTERN, r'\1????????????????', text)

    return text


if __name__ == '__main__':
    text = read_text_from_file("../text4.txt")
    print(text)
    masked_text = mask_bank_cards(text)
    print(masked_text)

