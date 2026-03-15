"""
Validate phone number using phonenumbers library. Also return information about the phone number such as country.
"""
import phonenumbers
import logging
from phonenumbers import NumberParseException, PhoneNumberFormat

logger = logging.getLogger(__name__)

def validate_phone_number(phone_number: str) -> tuple[bool, str]:
    """
    Given a phone number, return a tuple of (bool, str) where the bool is True if the phone number is valid
    and False otherwise.
    """
    phone_number = str(phone_number).strip() if phone_number is not None else ""
    if not phone_number or phone_number.lower() in ("nan", "none"):
        return False, "Empty or whitespace"
    try:
        parsed = phonenumbers.parse(phone_number, None)
        if phonenumbers.is_valid_number(parsed):
            region = phonenumbers.region_code_for_number(parsed)
            return True, region or "Unknown"
        return False, "Invalid number"
    except NumberParseException as e:
        return False, f"Parse error: {str(e)}"

def to_e164_if_valid(phone_number, region=None):
    """
    If the number is valid, return E.164 string; else None.
    """
    phone_number = str(phone_number).strip() if phone_number is not None else ""
    if not phone_number or phone_number.lower() in ("nan", "none"):
        return None
    try:
        parsed = phonenumbers.parse(phone_number, region)
        if phonenumbers.is_valid_number(parsed):
            return phonenumbers.format_number(parsed, PhoneNumberFormat.E164)
        return None
    except NumberParseException:
        return None

def try_with_region(phone_number: str, region_code: str) -> tuple[bool, str | None]:
    """
    Try parsing the phone number with the given region.
    """
    phone_number = str(phone_number).strip() if phone_number is not None else ""
    if not phone_number or phone_number.lower() in ("nan", "none") or not region_code:
        return False, None
    try:
        parsed = phonenumbers.parse(phone_number, region_code)
        if phonenumbers.is_valid_number(parsed):
            e164 = phonenumbers.format_number(parsed, PhoneNumberFormat.E164)
            return True, e164
        return False, None
    except NumberParseException:
        return False, None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_numbers = [
        "+1 650 253 0000",
        "+44 20 7946 0958",
        "+1 555 123 4567",
    ]
    for number in test_numbers:
        valid, details = validate_phone_number(number)
        logger.info(f"Number: {number!r} | Valid: {valid} | Details: {details}")