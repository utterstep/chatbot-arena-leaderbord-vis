#!/usr/bin/env python3
"""
Convert Elo rating pickle files to JSON format for web visualization.
"""
import pickle
import re
import json
import glob
from datetime import datetime
from pathlib import Path

import pandas as pd


def extract_date_from_filename(filename):
    """Extract date from filename in the format elo_results_YYYYMMDD.pkl"""
    match = re.search(r'elo_results_(\d{8})\.pkl', filename)
    if match:
        date_str = match.group(1)
        return datetime.strptime(date_str, '%Y%m%d')
    return None


def load_elo_data(file_path):
    """Load Elo data from a pickle file and extract max ratings for each category"""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        results = []

        # Check if the data has text, vision, and image keys
        if isinstance(data, dict):
            # Process text models
            if 'text' in data:
                for category, category_data in data['text'].items():
                    if 'elo_rating_final' in category_data:
                        elo_ratings = category_data['elo_rating_final']
                        if isinstance(elo_ratings, pd.Series):
                            results.append({
                                'date': extract_date_from_filename(file_path).strftime('%Y-%m-%d'),
                                'category': f'text_{category}',
                                'max_elo': float(elo_ratings.max()),
                                'max_model': elo_ratings.idxmax()
                            })

            # Process vision models
            if 'vision' in data:
                for category, category_data in data['vision'].items():
                    if 'elo_rating_final' in category_data:
                        elo_ratings = category_data['elo_rating_final']
                        if isinstance(elo_ratings, pd.Series):
                            results.append({
                                'date': extract_date_from_filename(file_path).strftime('%Y-%m-%d'),
                                'category': f'vision_{category}',
                                'max_elo': float(elo_ratings.max()),
                                'max_model': elo_ratings.idxmax()
                            })

            # Process image models
            if 'image' in data:
                for category, category_data in data['image'].items():
                    if 'elo_rating_final' in category_data:
                        elo_ratings = category_data['elo_rating_final']
                        if isinstance(elo_ratings, pd.Series):
                            results.append({
                                'date': extract_date_from_filename(file_path).strftime('%Y-%m-%d'),
                                'category': f'image_{category}',
                                'max_elo': float(elo_ratings.max()),
                                'max_model': elo_ratings.idxmax()
                            })

        return results
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []


def main():
    """Convert all pickle files to a single JSON file"""
    all_files = sorted(glob.glob('elo_results_*.pkl'))

    if not all_files:
        print("No elo_results_*.pkl files found in current directory")
        return

    all_data = []

    for file_path in all_files:
        results = load_elo_data(file_path)
        all_data.extend(results)

    # Sort data by date
    all_data.sort(key=lambda x: x['date'])

    # Save to JSON file
    with open('elo_data.json', 'w') as f:
        json.dump(all_data, f, indent=2)

    print(f"Converted {len(all_data)} records to elo_data.json")


if __name__ == "__main__":
    main()
