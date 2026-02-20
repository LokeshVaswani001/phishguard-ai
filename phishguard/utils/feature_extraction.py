import re
from urllib.parse import urlparse


def having_ip(url):
    pattern = r'(([0-9]{1,3}\.){3}[0-9]{1,3})'
    return 1 if re.search(pattern, url) else 0


def having_at_symbol(url):
    return 1 if "@" in url else 0


def url_length(url):
    return 1 if len(url) > 75 else 0


def url_depth(url):
    path = urlparse(url).path
    return path.count('/')


def redirection(url):
    return 1 if url.count('//') > 1 else 0


def https_domain(url):
    return 1 if "https" in url else 0


def tiny_url(url):
    shortening_services = r"bit\.ly|goo\.gl|tinyurl|ow\.ly|t\.co"
    return 1 if re.search(shortening_services, url) else 0


def prefix_suffix(url):
    return 1 if '-' in urlparse(url).netloc else 0


def extract_features(url):
    features = []

    features.append(having_ip(url))
    features.append(having_at_symbol(url))
    features.append(url_length(url))
    features.append(url_depth(url))
    features.append(redirection(url))
    features.append(https_domain(url))
    features.append(tiny_url(url))
    features.append(prefix_suffix(url))

    # Remaining dummy features to match training structure
    while len(features) < 16:
        features.append(0)

    return features
