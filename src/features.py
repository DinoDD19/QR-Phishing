import urllib.parse
import tldextract


def extract_features(url: str):
    """Extract features from a URL for classification."""
    parsed = urllib.parse.urlparse(url)
    ext = tldextract.extract(url)

    return [
        len(url),                              # URL length
        url.count("-"),                        # Hyphen count
        url.count("."),                        # Dot count
        url.count("/"),                        # Slash count
        len(ext.domain),                       # Domain length
        1 if parsed.scheme == "https" else 0   # HTTPS presence
    ]
