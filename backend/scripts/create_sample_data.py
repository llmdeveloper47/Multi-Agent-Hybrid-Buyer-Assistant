#!/usr/bin/env python3
"""
Create Sample Data Script

Generates a small sample dataset for testing the Buyer Advisor system
without needing the full Kaggle dataset.

Usage:
    python -m scripts.create_sample_data --output ./data/sample_properties.csv
"""

import argparse
import random
import os
from pathlib import Path

import pandas as pd


# Sample data for generating properties
CITIES = {
    "CA": ["Los Angeles", "San Francisco", "San Diego", "Sacramento", "Oakland", "San Jose"],
    "TX": ["Austin", "Houston", "Dallas", "San Antonio", "Fort Worth", "El Paso"],
    "NY": ["New York", "Buffalo", "Rochester", "Albany", "Syracuse"],
    "FL": ["Miami", "Orlando", "Tampa", "Jacksonville", "Fort Lauderdale"],
    "WA": ["Seattle", "Tacoma", "Spokane", "Bellevue", "Kirkland"],
    "CO": ["Denver", "Colorado Springs", "Boulder", "Aurora"],
    "AZ": ["Phoenix", "Scottsdale", "Tucson", "Mesa"],
    "IL": ["Chicago", "Aurora", "Naperville", "Joliet"],
    "GA": ["Atlanta", "Savannah", "Augusta", "Athens"],
    "NC": ["Charlotte", "Raleigh", "Durham", "Greensboro"],
}

PROPERTY_TYPES = [
    "Single Family",
    "Condo",
    "Townhouse",
    "Multi-Family",
    "Apartment",
]

STREET_NAMES = [
    "Main", "Oak", "Maple", "Cedar", "Pine", "Elm", "Birch",
    "Park", "Lake", "Hill", "Valley", "River", "Mountain",
    "First", "Second", "Third", "Fourth", "Fifth",
    "Washington", "Lincoln", "Jefferson", "Franklin", "Madison",
]

STREET_TYPES = ["St", "Ave", "Blvd", "Dr", "Ln", "Way", "Ct", "Rd"]


def generate_address():
    """Generate a random street address."""
    number = random.randint(100, 9999)
    street = random.choice(STREET_NAMES)
    street_type = random.choice(STREET_TYPES)
    return f"{number} {street} {street_type}"


def generate_zip(state: str) -> str:
    """Generate a plausible ZIP code for a state."""
    zip_prefixes = {
        "CA": ["90", "91", "92", "93", "94", "95"],
        "TX": ["75", "76", "77", "78", "79"],
        "NY": ["10", "11", "12", "13", "14"],
        "FL": ["32", "33", "34"],
        "WA": ["98", "99"],
        "CO": ["80", "81"],
        "AZ": ["85", "86"],
        "IL": ["60", "61", "62"],
        "GA": ["30", "31"],
        "NC": ["27", "28"],
    }
    prefix = random.choice(zip_prefixes.get(state, ["00"]))
    suffix = str(random.randint(0, 999)).zfill(3)
    return prefix + suffix


def generate_property(idx: int) -> dict:
    """Generate a single random property."""
    state = random.choice(list(CITIES.keys()))
    city = random.choice(CITIES[state])
    prop_type = random.choice(PROPERTY_TYPES)
    
    # Base price varies by location
    base_prices = {
        "CA": 800000, "NY": 700000, "WA": 650000,
        "CO": 550000, "FL": 450000, "TX": 400000,
        "AZ": 400000, "IL": 350000, "GA": 350000, "NC": 350000,
    }
    base_price = base_prices.get(state, 400000)
    
    # Adjust by property type
    type_multipliers = {
        "Single Family": 1.0,
        "Condo": 0.7,
        "Townhouse": 0.85,
        "Multi-Family": 1.3,
        "Apartment": 0.6,
    }
    
    bedrooms = random.choices(
        [1, 2, 3, 4, 5, 6],
        weights=[0.05, 0.2, 0.35, 0.25, 0.1, 0.05]
    )[0]
    
    bathrooms = random.choices(
        [1, 1.5, 2, 2.5, 3, 3.5, 4],
        weights=[0.1, 0.15, 0.3, 0.2, 0.15, 0.05, 0.05]
    )[0]
    
    # Size based on bedrooms
    base_sqft = 800 + (bedrooms * 300)
    size = int(base_sqft * random.uniform(0.8, 1.4))
    
    # Calculate price
    price = base_price * type_multipliers.get(prop_type, 1.0)
    price *= (bedrooms / 3)  # Adjust for bedrooms
    price *= random.uniform(0.7, 1.4)  # Random variation
    price = int(round(price, -3))  # Round to nearest thousand
    
    return {
        "Type": prop_type,
        "Price": price,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Size": size,
        "City": city,
        "State": state,
        "Address": generate_address(),
        "Zip": generate_zip(state),
    }


def create_sample_dataset(num_records: int = 1000) -> pd.DataFrame:
    """
    Create a sample dataset of properties.
    
    Args:
        num_records: Number of records to generate
        
    Returns:
        DataFrame with sample properties
    """
    properties = [generate_property(i) for i in range(num_records)]
    return pd.DataFrame(properties)


def main():
    parser = argparse.ArgumentParser(
        description="Generate sample real estate data for testing"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/sample_properties.csv",
        help="Output path for the CSV file"
    )
    parser.add_argument(
        "--num-records",
        type=int,
        default=1000,
        help="Number of records to generate (default: 1000)"
    )
    
    args = parser.parse_args()
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {args.num_records} sample properties...")
    df = create_sample_dataset(args.num_records)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Sample data saved to {output_path}")
    
    # Show sample
    print("\nSample records:")
    print(df.head(10).to_string())
    
    print(f"\nDataset statistics:")
    print(f"  Total records: {len(df)}")
    print(f"  States covered: {df['State'].nunique()}")
    print(f"  Cities covered: {df['City'].nunique()}")
    print(f"  Price range: ${df['Price'].min():,.0f} - ${df['Price'].max():,.0f}")
    print(f"  Avg price: ${df['Price'].mean():,.0f}")


if __name__ == "__main__":
    main()

