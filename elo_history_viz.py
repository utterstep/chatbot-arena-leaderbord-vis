#!/usr/bin/env python3
"""
Interactive visualization of max Elo ratings over time from the Chatbot Arena leaderboard data.
"""
import os
import re
import pickle
import glob
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from tqdm import tqdm


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
        
        results = {}
        
        # Check if the data has text, vision, and image keys
        if isinstance(data, dict):
            # Process text models
            if 'text' in data:
                for category, category_data in data['text'].items():
                    if 'elo_rating_final' in category_data:
                        elo_ratings = category_data['elo_rating_final']
                        if isinstance(elo_ratings, pd.Series):
                            results[f'text_{category}'] = {
                                'max_elo': elo_ratings.max(),
                                'max_model': elo_ratings.idxmax()
                            }
            
            # Process vision models
            if 'vision' in data:
                for category, category_data in data['vision'].items():
                    if 'elo_rating_final' in category_data:
                        elo_ratings = category_data['elo_rating_final']
                        if isinstance(elo_ratings, pd.Series):
                            results[f'vision_{category}'] = {
                                'max_elo': elo_ratings.max(),
                                'max_model': elo_ratings.idxmax()
                            }
            
            # Process image models
            if 'image' in data:
                for category, category_data in data['image'].items():
                    if 'elo_rating_final' in category_data:
                        elo_ratings = category_data['elo_rating_final']
                        if isinstance(elo_ratings, pd.Series):
                            results[f'image_{category}'] = {
                                'max_elo': elo_ratings.max(),
                                'max_model': elo_ratings.idxmax()
                            }
        
        return results
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}


def collect_all_data(data_dir='.'):
    """Collect data from all pickle files and organize by date"""
    all_files = sorted(glob.glob(os.path.join(data_dir, 'elo_results_*.pkl')))
    
    if not all_files:
        raise ValueError(f"No elo_results_*.pkl files found in {data_dir}")
    
    all_data = []
    
    for file_path in tqdm(all_files, desc="Processing files"):
        date = extract_date_from_filename(file_path)
        if date:
            results = load_elo_data(file_path)
            
            for category, category_data in results.items():
                all_data.append({
                    'date': date,
                    'category': category,
                    'max_elo': category_data['max_elo'],
                    'max_model': category_data['max_model']
                })
    
    return pd.DataFrame(all_data)


def create_app(df):
    """Create a Dash app for interactive visualization"""
    app = dash.Dash(__name__, title="Chatbot Arena Max Elo History")
    
    # Get unique categories for dropdown
    categories = sorted(df['category'].unique())
    
    app.layout = html.Div([
        html.H1("Chatbot Arena - Maximum Elo Rating History"),
        
        html.Div([
            html.Label("Select Categories:"),
            dcc.Dropdown(
                id='category-dropdown',
                options=[{'label': cat, 'value': cat} for cat in categories],
                value=['text_full', 'text_coding'],
                multi=True
            ),
        ], style={'width': '50%', 'margin': '20px'}),
        
        dcc.Graph(id='elo-history-graph'),
        
        html.Div([
            html.H3("Models with Highest Elo Rating"),
            html.Div(id='top-models-info')
        ], style={'margin': '20px'}),
        
    ], style={'font-family': 'Arial, sans-serif', 'margin': '20px'})
    
    @app.callback(
        [Output('elo-history-graph', 'figure'),
         Output('top-models-info', 'children')],
        [Input('category-dropdown', 'value')]
    )
    def update_graph(selected_categories):
        if not selected_categories:
            return go.Figure(), html.P("No categories selected")
        
        # Filter data based on selected categories
        filtered_df = df[df['category'].isin(selected_categories)]
        
        # Create line chart for max Elo history
        fig = px.line(
            filtered_df, 
            x='date', 
            y='max_elo', 
            color='category',
            title='Maximum Elo Rating Over Time',
            labels={'date': 'Date', 'max_elo': 'Maximum Elo Rating', 'category': 'Category', 'max_model': 'Model'},
            markers=True,
            hover_data=['max_model']  # Show model name on hover
        )
        
        fig.update_layout(
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_title="Date",
            yaxis_title="Elo Rating",
            template="plotly_white"
        )
        
        # Create information about the latest top models
        latest_models = []
        for category in selected_categories:
            category_df = df[df['category'] == category]
            if not category_df.empty:
                latest_date = category_df['date'].max()
                latest_data = category_df[category_df['date'] == latest_date].iloc[0]
                latest_models.append(
                    html.Div([
                        html.H4(category),
                        html.P(f"Top model: {latest_data['max_model']}"),
                        html.P(f"Elo rating: {latest_data['max_elo']:.2f}"),
                        html.P(f"Date: {latest_data['date'].strftime('%Y-%m-%d')}"),
                    ], style={'margin-bottom': '20px'})
                )
        
        return fig, latest_models
    
    return app


def main():
    """Main function to run the visualization"""
    # Collect data from all pickle files
    try:
        data_df = collect_all_data()
        print(f"Collected data from {len(data_df)} records across {data_df['date'].nunique()} dates")
        
        # Create and run the Dash app
        app = create_app(data_df)
        print("Starting the Dash app. Press Ctrl+C to exit.")
        app.run(debug=True)
    
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
