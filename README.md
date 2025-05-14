---
title: Chatbot Arena Leaderboard
emoji: 🏆🤖
colorFrom: indigo
colorTo: green
sdk: gradio
pinned: false
license: apache-2.0
tags:
- leaderboard
sdk_version: 4.44.1
---

# Chatbot Arena Elo Rating History

Interactive visualization of maximum Elo ratings over time from the Chatbot Arena leaderboard data.

## Features

- Interactive line chart showing Elo rating history
- Category selection for different model types (text, vision, image)
- Hover information showing model details
- Top models information panel
- Responsive design

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Convert pickle data to JSON:
```bash
python convert_to_json.py
```

3. Serve the website locally:
```bash
python -m http.server
```

4. Open your browser and navigate to `http://localhost:8000`

## Data Format

The visualization expects pickle files in the format `elo_results_YYYYMMDD.pkl` in the root directory. These files should contain Elo rating data for different model categories.

## Deployment

The site is automatically deployed to GitHub Pages when changes are pushed to the main branch. The deployment process:

1. Converts pickle data to JSON format
2. Builds the static website
3. Deploys to GitHub Pages

You can also manually trigger the deployment from the Actions tab in GitHub.

## License

MIT License
