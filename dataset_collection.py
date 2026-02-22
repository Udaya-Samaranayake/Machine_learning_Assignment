"""
Dataset Collection Script - Sri Lankan Vegetable Prices
========================================================

Data Source: Department of Census and Statistics, Sri Lanka
URL: https://www.statistics.gov.lk/DashBoard/Prices/
Data Endpoint: https://www.statistics.gov.lk/DashBoard/Prices/Prices_Data.php

Descrition:
    This script collects weekly retail vegetable price data from the
    Sri Lankan Department of Census and Statistics. The data covers
    prices collected from 14 markets in the Colombo District.

    - Date range: January 2017 to January 2026 (~436 weeks)
    - 24 core vegetables + leaves + potatoes = 37 vegetable-related items
    - Prices are weekly averages in Sri Lankan Rupees (LKR)
    - Categories: Low Country Vegetables, Up Country Vegetables, Leaves, Potatoes

Output:
    - data/sri_lankan_vegetable_prices.csv  (wide format - one column per vegetable)
    - data/sri_lankan_vegetable_prices_long.csv  (long format - better for ML)
"""

import os
import re
import json
import urllib.request
import pandas as pd
import numpy as np
from datetime import datetime

DATA_URL = "https://www.statistics.gov.lk/DashBoard/Prices/Prices_Data.php"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")

VEGETABLE_CATEGORIES = [
    "Low Country Vegetables",
    "Up Country Vegetables",
    "Leaves",
    "Potatoes",
]

VEGETABLE_KEY_PATTERNS = ["LCVEG", "UPCVEG", "LEAVES", "POTATOES"]

EXCLUDE_PATTERNS = ["Betel_Leaves", "Arecanuts"]

def fetch_raw_data(url=DATA_URL):
    """Fetch raw JavaScript data from the statistics.gov.lk endpoint."""
    print(f"Fetching data from: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, timeout=120)
    raw = resp.read().decode("utf-8-sig")
    print(f"  Downloaded {len(raw):,} characters")
    return raw

def parse_product_info(raw_js):
    """Parse the product information array (pip) from the JavaScript."""
    match = re.search(r"var pip=\[(.*?)\];", raw_js, re.DOTALL)
    if not match:
        raise ValueError("Could not find 'pip' array in the data")
    pip = json.loads("[" + match.group(1) + "]")
    print(f"  Found {len(pip)} products")
    return pip


def parse_price_data(raw_js):
    """Parse the weekly price data array from the JavaScript."""
    start_marker = "var prices=["
    start_idx = raw_js.find(start_marker)
    if start_idx == -1:
        raise ValueError("Could not find 'prices' array in the data")

    idx = start_idx + len(start_marker)
    bracket_count = 1
    while bracket_count > 0 and idx < len(raw_js):
        if raw_js[idx] == "[":
            bracket_count += 1
        elif raw_js[idx] == "]":
            bracket_count -= 1
        idx += 1

    prices_str = raw_js[start_idx + len("var prices=") : idx]

    prices_str = prices_str.replace("''", "null")

    prices = json.loads(prices_str)
    print(f"  Found {len(prices)} weekly price records")
    return prices


def identify_vegetable_columns(prices, pip):
    """Identify columns that correspond to vegetables."""
    all_keys = list(prices[0].keys())
    veg_keys = []

    for key in all_keys:
        if key == "Date":
            continue
        is_veg = any(pat in key for pat in VEGETABLE_KEY_PATTERNS)
    
        is_excluded = any(exc in key for exc in EXCLUDE_PATTERNS)

        if is_veg and not is_excluded:
            veg_keys.append(key)

    print(f"  Identified {len(veg_keys)} vegetable columns")
    return veg_keys


def parse_date(date_str):
    """
    Parse date strings like 'W1.Jan.2017' into proper datetime objects.
    W1 = first week, W2 = second week, etc.
    """
    match = re.match(r"W(\d+)\.(\w+)\.(\d{4})", date_str)
    if not match:
        return None

    week_num = int(match.group(1))
    month_str = match.group(2)
    year = int(match.group(3))

    month_map = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
        "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
        "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
    }
    month = month_map.get(month_str)
    if month is None:
        return None
    try:
        day = min(1 + (week_num - 1) * 7, 28) 
        return datetime(year, month, day)
    except ValueError:
        return None

def create_product_name_map(pip):
    """Create a mapping from product keys to friendly names."""
    name_map = {}
    for p in pip:
        name_map[p["product"]] = p["name"].strip()
    return name_map

def build_wide_dataframe(prices, veg_keys, pip):
    """Build a wide-format DataFrame with one column per vegetable."""
    name_map = create_product_name_map(pip)

    rows = []
    for record in prices:
        row = {
            "date_raw": record["Date"],
            "date": parse_date(record["Date"]),
        }
        for key in veg_keys:
            friendly_name = name_map.get(key, key)
            row[friendly_name] = record.get(key)
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("date").reset_index(drop=True)

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    week_series = df["date"].dt.isocalendar().week
    df["week_of_year"] = pd.to_numeric(week_series, errors="coerce").astype("Int64")

    return df


def build_long_dataframe(df_wide, veg_keys, pip):
    """Convert wide-format DataFrame to long format (better for ML)."""
    name_map = create_product_name_map(pip)

    friendly_names = [name_map.get(k, k) for k in veg_keys]

    cat_map = {}
    for p in pip:
        cat_map[p["name"].strip()] = p["category"]

    subcat_map = {}
    for k in veg_keys:
        if "LCVEG" in k:
            subcat_map[name_map.get(k, k)] = "Low Country"
        elif "UPCVEG" in k:
            subcat_map[name_map.get(k, k)] = "Up Country"
        elif "LEAVES" in k:
            subcat_map[name_map.get(k, k)] = "Leaves"
        elif "POTATOES" in k:
            subcat_map[name_map.get(k, k)] = "Potatoes"

    df_long = df_wide.melt(
        id_vars=["date_raw", "date", "year", "month", "week_of_year"],
        value_vars=friendly_names,
        var_name="vegetable",
        value_name="price_lkr",
    )

    df_long["category"] = df_long["vegetable"].map(cat_map)
    df_long["sub_category"] = df_long["vegetable"].map(subcat_map)

    df_long = df_long.sort_values(["date", "vegetable"]).reset_index(drop=True)

    return df_long

def collect_dataset():
    """Main function to collect and save the vegetable price dataset."""
    print("=" * 60)
    print("Sri Lankan Vegetable Price Dataset Collection")
    print("=" * 60)

    print("\n[1/5] Fetching raw data from statistics.gov.lk...")
    raw_js = fetch_raw_data()

    print("\n[2/5] Parsing product information...")
    pip = parse_product_info(raw_js)

    print("\n[3/5] Parsing weekly price data...")
    prices = parse_price_data(raw_js)

    print("\n[4/5] Processing vegetable data...")
    veg_keys = identify_vegetable_columns(prices, pip)

    df_wide = build_wide_dataframe(prices, veg_keys, pip)
    df_long = build_long_dataframe(df_wide, veg_keys, pip)

    print("\n[5/5] Saving datasets...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    wide_path = os.path.join(OUTPUT_DIR, "sri_lankan_vegetable_prices.csv")
    long_path = os.path.join(OUTPUT_DIR, "sri_lankan_vegetable_prices_long.csv")

    df_wide.to_csv(wide_path, index=False)
    df_long.to_csv(long_path, index=False)

    print(f"  Wide format: {wide_path}")
    print(f"    Shape: {df_wide.shape}")
    print(f"  Long format: {long_path}")
    print(f"    Shape: {df_long.shape}")

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"  Data Source: Department of Census & Statistics, Sri Lanka")
    print(f"  URL: https://www.statistics.gov.lk/DashBoard/Prices/")
    print(f"  Date Range: {df_wide['date'].min().strftime('%Y-%m-%d')} to "
          f"{df_wide['date'].max().strftime('%Y-%m-%d')}")
    print(f"  Total Weeks: {len(df_wide)}")
    print(f"  Vegetables: {len(veg_keys)}")
    print(f"  Total Price Records (long format): {len(df_long)}")
    print(f"  Missing Values: {df_long['price_lkr'].isna().sum()} "
          f"({df_long['price_lkr'].isna().mean()*100:.1f}%)")

    print("\n  Vegetables included:")
    name_map = create_product_name_map(pip)
    for key in veg_keys:
        name = name_map.get(key, key)
        non_null = df_long[df_long["vegetable"] == name]["price_lkr"].notna().sum()
        print(f"    - {name}: {non_null} weeks of data")

    print("\n  Price Statistics (LKR per kg/unit):")
    stats = df_long.groupby("vegetable")["price_lkr"].agg(["mean", "min", "max", "std"])
    stats = stats.round(2)
    print(stats.to_string())

    return df_wide, df_long


if __name__ == "__main__":
    df_wide, df_long = collect_dataset()
