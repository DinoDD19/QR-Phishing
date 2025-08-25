import urllib.parse

def extract_features(url):
    parsed_url = urllib.parse.urlparse(url)
    return {
        "length": len(url),  # URL length
        "num_digits": sum(c.isdigit() for c in url),  # Number of digits in URL
        "num_special_chars": sum(c in "!@#$%^&*()_+" for c in url),  # Special characters count
        "domain_length": len(parsed_url.netloc),  # Domain length
        "path_length": len(parsed_url.path),  # Path length
    }

# Apply the function to all URLs
features = df["url"].apply(extract_features).apply(pd.Series)

# Add labels (1 = phishing, 0 = safe)
features["label"] = 1  # Since our dataset is phishing, we mark all as 1

# Show first 5 rows
features.head()
