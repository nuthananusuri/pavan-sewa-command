"""
Victim-data protection.

* Reporter name and phone are encrypted at rest with AES (Fernet:
  AES-128-CBC + HMAC-SHA256). On IBM Z / LinuxONE, OpenSSL routes AES
  through the CPACF on-chip crypto instructions, so this costs almost
  nothing - and it sits on top of the platform's pervasive encryption.
* Dashboards and logs only ever see a keyed pseudonym (HMAC-SHA256) and a
  masked phone number. Decryption is a separate, audited call that only
  the "rescue-lead" role may make.
"""
import hashlib
import hmac
import os

try:
    from cryptography.fernet import Fernet
except ImportError:  # pragma: no cover
    Fernet = None

KEY_FILE = os.environ.get("PSC_KEY_FILE", os.path.join(os.path.dirname(__file__), "..", ".psc_key"))


def _load_key():
    env = os.environ.get("PSC_DATA_KEY")
    if env:
        return env.encode()
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, "rb") as f:
            return f.read().strip()
    if Fernet is None:
        raise RuntimeError("install 'cryptography' (apt: python3-cryptography)")
    key = Fernet.generate_key()
    fd = os.open(KEY_FILE, os.O_WRONLY | os.O_CREAT, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(key)
    return key


class Vault:
    def __init__(self):
        key = _load_key()
        self._fernet = Fernet(key)
        self._hkey = hashlib.sha256(b"pseudonym:" + key).digest()

    def encrypt(self, text):
        return self._fernet.encrypt(str(text).encode()).decode()

    def decrypt(self, token):
        return self._fernet.decrypt(token.encode()).decode()

    def pseudonym(self, text):
        return "V-" + hmac.new(self._hkey, str(text).encode(), hashlib.sha256).hexdigest()[:10].upper()

    @staticmethod
    def mask_phone(phone):
        p = str(phone)
        return p[:2] + "*" * max(0, len(p) - 4) + p[-2:]
