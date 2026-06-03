import json
import pandas as pd
from pandas import json_normalize

with open("USAspending-data-catalog.json") as f:
    data = json.load(f)

datasets = data["dataset"]
print(f"Loaded {len(datasets)} datasets\n")

df = json_normalize(datasets)
print("Columns:", df.columns.tolist())
print("\nShape:", df.shape)

print("\n── Dataset titles ──")
print(df["title"].to_string(index=False))

print("\n── Publishers ──")
print(df["publisher.name"].value_counts().to_string())

print("\n── Access levels ──")
print(df["accessLevel"].value_counts().to_string())

print("\n── Update frequencies (accrualPeriodicity) ──")
print(df["accrualPeriodicity"].value_counts().to_string())

print("\n── Distribution formats ──")
formats = [d[0]["format"] for d in df["distribution"] if d]
print(pd.Series(formats).value_counts().to_string())

print("\n── Access URLs ──")
for _, row in df.iterrows():
    dist = row["distribution"]
    url = dist[0]["accessURL"] if dist else "N/A"
    print(f"  {row['title']}: {url}")

print("\n── Top 20 keywords across all datasets ──")
keyword_counts = df["keyword"].explode().value_counts()
print(keyword_counts.head(20).to_string())

print("\n── Keywords per dataset ──")
for _, row in df.iterrows():
    print(f"  {row['title']}: {len(row['keyword'])} keywords")

print("\n── Shared keywords between datasets ──")
kw_sets = {row["title"]: set(row["keyword"]) for _, row in df.iterrows()}
titles = list(kw_sets.keys())
for i in range(len(titles)):
    for j in range(i + 1, len(titles)):
        shared = kw_sets[titles[i]] & kw_sets[titles[j]]
        if shared:
            print(f"  {titles[i]} ∩ {titles[j]}: {len(shared)} shared")

print("\n── Contact points ──")
for _, row in df.iterrows():
    cp = row.get("contactPoint", {})
    name = cp.get("fn", "N/A")
    email = cp.get("hasEmail", "N/A").replace("mailto:", "")
    print(f"  {row['title']}: {name} ({email})")

print("\n── Download URL availability ──")
for _, row in df.iterrows():
    dist = row["distribution"]
    has_download = bool(dist and dist[0].get("downloadURL"))
    print(f"  {row['title']}: {'✓ has downloadURL' if has_download else '✗ no downloadURL'}")
