import pandas as pd
import random
import tldextract

# Load the dataset
file_path = r"C:\AWS\qr-code\dataseturl.xlsx"  # Path to your original dataset
df = pd.read_excel(file_path)

# Extract safe URLs (assuming they are in the first column)
safe_urls = df.iloc[:, 0].dropna().unique().tolist()

# Generate fake URLs
fake_urls = []
for url in safe_urls:
    fake_url = f"https://{random.randint(1000, 9999)}-{url}"  # Example: https://1234-google.com
    fake_urls.append(fake_url)

# Create a labeled dataset
data = pd.DataFrame({
    "url": safe_urls + fake_urls,
    "label": [1] * len(safe_urls) + [0] * len(fake_urls)  # 1 = safe, 0 = fake
})

# Feature extraction
def extract_features(url):
    ext = tldextract.extract(url)
    return [
        len(url),                       # URL length
        url.count('-'),                 # Hyphen count
        url.count('.'),                 # Dot count
        url.count('/'),                 # Slash count
        len(ext.domain),                # Domain length
        1 if "https" in url else 0      # HTTPS presence
    ]

data["features"] = data["url"].apply(extract_features)

# Save the dataset
data.to_csv("labeled_urls.csv", index=False)
print("✅ Dataset saved as labeled_urls.csv")