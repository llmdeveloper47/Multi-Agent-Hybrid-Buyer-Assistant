#!/usr/bin/env python3
"""
Data Ingestion Script for Hybrid Buyer Advisor (Superlinked Version)

This script loads the realtor dataset and ingests it into Superlinked.

Usage:
    python -m scripts.data_ingestion_superlinked --data-path ./data/realtor-data.csv
    
Options:
    --data-path: Path to the CSV file (required)
    --sample-size: Number of records to sample (optional, for testing)
    --batch-size: Batch size for ingestion (default: 1000)
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.services.superlinked_service import get_superlinked_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_and_clean_data(
    data_path: str,
    sample_size: Optional[int] = None
) -> pd.DataFrame:
    """
    Load and clean the real estate dataset.
    
    Args:
        data_path: Path to CSV file
        sample_size: Optional sample size for testing
        
    Returns:
        Cleaned DataFrame
    """
    logger.info(f"Loading data from {data_path}...")
    
    # Load CSV
    df = pd.read_csv(data_path, low_memory=False)
    logger.info(f"Loaded {len(df)} records")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Clean data - drop rows with missing critical values
    critical_columns = ['price', 'city', 'state']
    existing_critical = [col for col in critical_columns if col in df.columns]
    
    if existing_critical:
        initial_count = len(df)
        df = df.dropna(subset=existing_critical)
        logger.info(f"Dropped {initial_count - len(df)} rows with missing critical values")
    
    # Filter out invalid prices
    if 'price' in df.columns:
        df = df[df['price'] > 0]
        df = df[df['price'] < 100_000_000]
        logger.info(f"After price filtering: {len(df)} records")
    
    # Sample if requested
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
        logger.info(f"Sampled {sample_size} records")
    
    # Log statistics
    logger.info(f"States in dataset: {df['state'].nunique()}")
    logger.info(f"Cities in dataset: {df['city'].nunique()}")
    logger.info(f"Price range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")
    
    return df


def prepare_properties(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Prepare properties for Superlinked ingestion.
    
    Maps exactly to realtor-data.csv columns:
    brokered_by, status, price, bed, bath, acre_lot, street, city, state, zip_code, house_size, prev_sold_date
    
    Args:
        df: DataFrame with property data
        
    Returns:
        List of property dictionaries
    """
    properties = []
    
    def safe_float(val, default=0.0):
        """Safely convert to float, handling NaN."""
        try:
            if pd.isna(val):
                return default
            return float(val)
        except (ValueError, TypeError):
            return default
    
    def safe_int(val, default=0):
        """Safely convert to int, handling NaN."""
        try:
            if pd.isna(val):
                return default
            return int(float(val))
        except (ValueError, TypeError):
            return default
    
    def safe_str(val, default=""):
        """Safely convert to string, handling NaN."""
        if pd.isna(val):
            return default
        return str(val)
    
    for idx, row in df.iterrows():
        prop = {
            # ID from DataFrame index
            "id": str(idx),
            
            # Numeric fields (exact CSV column names)
            "price": safe_float(row.get("price")),
            "bed": safe_int(row.get("bed")),
            "bath": safe_float(row.get("bath")),
            "house_size": safe_float(row.get("house_size")),
            "acre_lot": safe_float(row.get("acre_lot")),
            "zip_code": safe_int(row.get("zip_code")),
            
            # String fields (exact CSV column names)
            "status": safe_str(row.get("status"), "for_sale"),
            "city": safe_str(row.get("city")),
            "state": safe_str(row.get("state")),
            "street": safe_str(row.get("street")),
            "brokered_by": safe_str(row.get("brokered_by")),
            "prev_sold_date": safe_str(row.get("prev_sold_date")),
        }
        properties.append(prop)
    
    return properties


def ingest_data(
    properties: List[Dict[str, Any]],
    batch_size: int = 1000
) -> int:
    """
    Ingest properties into Superlinked.
    
    Args:
        properties: List of property dictionaries
        batch_size: Batch size for ingestion
        
    Returns:
        Number of records ingested
    """
    logger.info("Initializing Superlinked service...")
    
    superlinked = get_superlinked_service()
    
    logger.info(f"Starting ingestion of {len(properties)} records...")
    
    total_ingested = 0
    
    # Process in batches
    for i in tqdm(range(0, len(properties), batch_size), desc="Ingesting"):
        batch = properties[i:i + batch_size]
        
        try:
            count = superlinked.ingest_properties(batch)
            total_ingested += count
        except Exception as e:
            logger.error(f"Error ingesting batch {i}: {e}")
    
    logger.info(f"Ingestion complete. Total records ingested: {total_ingested}")
    
    return total_ingested


def verify_ingestion():
    """Verify the ingestion by running test queries."""
    logger.info("Verifying ingestion with Superlinked...")
    
    superlinked = get_superlinked_service(load_stats_from_data=False)
    
    # Get stats
    stats = superlinked.get_stats()
    logger.info(f"Superlinked stats: {stats}")
    
    # Run test searches using natural language
    test_queries = [
        "3 bedroom house under $300000",
        "4 bed 2 bath home in Florida",
        "large house with big lot in Texas",
        "affordable condo in Miami",
    ]
    
    for query in test_queries:
        logger.info(f"\nTest query: '{query}'")
        
        try:
            results = superlinked.search(
                query=query,
                top_k=3,
                use_natural_language=True
            )
            
            logger.info(f"Found {len(results)} results:")
            
            for i, result in enumerate(results, 1):
                # Use exact CSV column names
                city = result.get('city', 'N/A')
                state = result.get('state', 'N/A')
                price = result.get('price', 0)
                beds = result.get('bed', 'N/A')
                baths = result.get('bath', 'N/A')
                sqft = result.get('house_size', 'N/A')
                acre = result.get('acre_lot', 'N/A')
                status = result.get('status', 'N/A')
                score = result.get('score', 0)
                
                price_str = f"${price:,.0f}" if price else "N/A"
                sqft_str = f"{sqft:,.0f} sqft" if sqft and sqft != 'N/A' else "N/A"
                acre_str = f"{acre:.2f} acre" if acre and acre != 'N/A' else "N/A"
                
                logger.info(
                    f"  {i}. {city}, {state} - {price_str} | "
                    f"{beds} bed, {baths} bath | {sqft_str} | {acre_str} | "
                    f"status: {status} | score: {score:.3f}"
                )
                
        except Exception as e:
            logger.error(f"Error running query: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest real estate data into Superlinked"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the CSV data file"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of records to sample (optional)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for ingestion (default: 1000)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run verification after ingestion"
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.data_path):
        logger.error(f"Data file not found: {args.data_path}")
        sys.exit(1)
    
    # Load and clean data
    df = load_and_clean_data(args.data_path, args.sample_size)
    
    # Prepare properties
    properties = prepare_properties(df)
    
    # Ingest data
    total = ingest_data(properties, args.batch_size)
    
    # Verify if requested
    if args.verify and total > 0:
        verify_ingestion()
    
    logger.info("Done!")


if __name__ == "__main__":
    main()

