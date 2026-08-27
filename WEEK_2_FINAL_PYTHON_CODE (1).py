"""
Week 2 — Exploratory Data Analysis and Visualization
Dataset: New York City Airbnb Open Data (2019)

Place AB_NYC_2019.csv in the same folder before running.
The script creates charts in week2_outputs/.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA_FILE = Path("AB_NYC_2019.csv")
OUT = Path("week2_outputs")
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA_FILE)
print("Shape:", df.shape)
print(df.head())
print(df.info())
print(df.describe(include="all").T)

# Basic quality checks
print("\nMissing values:")
print(df.isna().sum().sort_values(ascending=False))
print("\nDuplicates:", df.duplicated().sum())

# Clean types used in EDA
df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df["minimum_nights"] = pd.to_numeric(df["minimum_nights"], errors="coerce")
df["number_of_reviews"] = pd.to_numeric(df["number_of_reviews"], errors="coerce")
df["reviews_per_month"] = pd.to_numeric(df["reviews_per_month"], errors="coerce")
df["availability_365"] = pd.to_numeric(df["availability_365"], errors="coerce")

# EDA transformation: exclude impossible zero prices for price-based analysis
price_df = df[df["price"] > 0].copy()
price_df["log_price"] = np.log1p(price_df["price"])

# 1. Listings by neighbourhood group
plt.figure(figsize=(8,5))
order = df["neighbourhood_group"].value_counts().index
sns.countplot(data=df, x="neighbourhood_group", order=order)
plt.title("Airbnb Listings by NYC Neighbourhood Group")
plt.xlabel("Neighbourhood group")
plt.ylabel("Number of listings")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(OUT/"01_listings_by_borough.png", dpi=180)
plt.close()

# 2. Room type distribution
plt.figure(figsize=(8,5))
order = df["room_type"].value_counts().index
sns.countplot(data=df, x="room_type", order=order)
plt.title("Airbnb Listings by Room Type")
plt.xlabel("Room type")
plt.ylabel("Number of listings")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(OUT/"02_listings_by_room_type.png", dpi=180)
plt.close()

# 3. Price distribution, using log scale for visibility
plt.figure(figsize=(8,5))
sns.histplot(price_df["price"], bins=60)
plt.xlim(0, 1000)
plt.title("Distribution of Airbnb Nightly Prices (≤ $1,000)")
plt.xlabel("Price (USD per night)")
plt.ylabel("Number of listings")
plt.tight_layout()
plt.savefig(OUT/"03_price_distribution.png", dpi=180)
plt.close()

# 4. Average price by borough
avg_borough = (
    price_df.groupby("neighbourhood_group")["price"]
    .mean().sort_values(ascending=False)
)
plt.figure(figsize=(8,5))
sns.barplot(x=avg_borough.index, y=avg_borough.values)
plt.title("Average Nightly Price by NYC Borough")
plt.xlabel("Neighbourhood group")
plt.ylabel("Average price (USD)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(OUT/"04_average_price_borough.png", dpi=180)
plt.close()

# 5. Average price by room type
avg_room = (
    price_df.groupby("room_type")["price"]
    .mean().sort_values(ascending=False)
)
plt.figure(figsize=(8,5))
sns.barplot(x=avg_room.index, y=avg_room.values)
plt.title("Average Nightly Price by Room Type")
plt.xlabel("Room type")
plt.ylabel("Average price (USD)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(OUT/"05_average_price_room_type.png", dpi=180)
plt.close()

# 6. Price vs room type by borough; cap display at 1000 to make patterns readable
plot_df = price_df[price_df["price"] <= 1000]
plt.figure(figsize=(11,6))
sns.boxplot(data=plot_df, x="room_type", y="price", hue="neighbourhood_group")
plt.title("Price Distribution by Room Type and NYC Borough (≤ $1,000)")
plt.xlabel("Room type")
plt.ylabel("Price (USD per night)")
plt.legend(title="Neighbourhood group", bbox_to_anchor=(1.02,1), loc="upper left")
plt.tight_layout()
plt.savefig(OUT/"06_price_room_borough_boxplot.png", dpi=180)
plt.close()

# 7. Reviews vs price using log price to reduce skew
sample = price_df.sample(min(10000, len(price_df)), random_state=42)
plt.figure(figsize=(9,6))
sns.scatterplot(data=sample, x="number_of_reviews", y="log_price", alpha=0.35)
plt.title("Reviews vs Log Price")
plt.xlabel("Number of reviews")
plt.ylabel("log(1 + price)")
plt.tight_layout()
plt.savefig(OUT/"07_reviews_vs_log_price.png", dpi=180)
plt.close()

# 8. Correlation heatmap for selected numeric fields
numeric = [
    "price","minimum_nights","number_of_reviews",
    "reviews_per_month","calculated_host_listings_count",
    "availability_365"
]
corr = df[numeric].corr(numeric_only=True)
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Matrix of Selected Numeric Variables")
plt.tight_layout()
plt.savefig(OUT/"08_correlation_heatmap.png", dpi=180)
plt.close()

# 9. Availability by room type
plt.figure(figsize=(9,6))
sns.boxplot(data=df, x="room_type", y="availability_365")
plt.title("Availability by Room Type")
plt.xlabel("Room type")
plt.ylabel("Available days (0–365)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(OUT/"09_availability_room_type.png", dpi=180)
plt.close()

# Aggregations for reporting
print("\nListings by borough:")
print(df["neighbourhood_group"].value_counts())

print("\nListings by room type:")
print(df["room_type"].value_counts())

print("\nAverage price by borough:")
print(price_df.groupby("neighbourhood_group")["price"].mean().sort_values(ascending=False))

print("\nAverage price by room type:")
print(price_df.groupby("room_type")["price"].mean().sort_values(ascending=False))

print("\nMedian price by borough:")
print(price_df.groupby("neighbourhood_group")["price"].median().sort_values(ascending=False))

print("\nCorrelation matrix:")
print(corr)

print("\nEDA completed. Charts saved to:", OUT)
