# Naruto NLP Project: Chatbot, Text Classification, character network visualization, and theme classification

This repository contains a comprehensive NLP project inspired by the Naruto series. The project explores various natural language processing tasks such as chatbot development, text classification, character network visualization, and theme classification. The data for this project is collected from multiple sources, including web scraping and Kaggle datasets.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)

## Introduction

The Naruto NLP Project brings the Naruto universe to life through natural language processing. It features:

- Naruto Chatbot: Chat interactively with Naruto using a fine-tuned conversational model.
- Text Classification: Categorize jutsu into types such as Genjutsu, Ninjutsu, and Taijutsu.
- Character Network Visualization: Build and explore character networks using named entity recognition (NER) and graph-based methods.
- Theme Classification: Identify themes in dialogues or scenes such as friendship, hope, sacrifice, and betrayal using zero-shot classification.
  This project utilizes data sourced through web scraping, Kaggle, and fandom APIs to create an immersive NLP experience.

## Dataset

The datasets and resources used in this project include:

- [Naruto Season 1 Subtitles](https://srtzilla.com/subtitle/naruto-season-1/english/2206507)
- [Naruto Episode 1 Transcript on Kaggle](https://www.kaggle.com/datasets/leonzatrax/naruto-ep-1-transcript)
- [Naruto Fandom Jutsu Database](https://naruto.fandom.com/wiki/Special:BrowseData/Jutsu?limit=250&offset=0&_cat=Jutsu)

### Data Collection

- Web Scraping: Subtitles and Jutsu data collected using Scrapy and BeautifulSoup.
- Kaggle API: Dataset programmatically accessed via Kaggle API.

## Installation

To set up the project, follow these steps:

1. Clone the repository to your local machine:

```
   git clone https://github.com/srijosh/NLP_Series_Analysis.git
```

2. Navigate to the project directory:
   cd NLP_Series_Analysi

3. Install dependencies:

```
   pip install -r requirements.txt

```

4. Configure environment variables:

- Copy .env_sample to .env.
- Replace placeholders with your HuggingFace token and other required credentials.

5. Run the project using :

```
   gradio gradio_app.py

```

### Requirements

This project uses the following libraries:

numpy
pandas
seaborn
scikit-learn
scrapy
beautifulsoup4
spacy
huggingface_hub
nltk
gradio
pyvis
evaluate
python-dotenv
peft
trl
bitsandbytes
spacy-transformers
transformers
accelerate
sentence-transformers
