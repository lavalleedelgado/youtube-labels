# youtube-labels
Ta-Yun Yang & Patrick Lavallee Delgado \\
Candidates, MS Computational Analysis and Public Policy

## Recovering Self-selected YouTube Video Categories
We attempt to recreate the decision a YouTube user makes selecting a category for his video using other instances of self-expression in the same content. These observations are encoded in language, including the title, description, tags, and caption of the video. This repository documents the data and modeling pipeline we use in our analysis.

## Requirements
- altair 4.1.0
- numpy 1.16.4      
- pandas 1.0.4
- PyYAML 5.1.1 
- scikit-learn 0.21.3
- spacy 2.2.4
- torch 1.4.0       
- torchtext 0.6.0
- youtube-transcript-api 0.3.1

## Structure
- data: video and category data by country as well as caption data for videos in the US data.
- exploration.ipynb: notebook that collects the counts of videos by category and the intersections of their vocabularies.
- get_captions.py: script that scrapes the caption data from the video page on the YouTube website.
- models_w_captions: results of models on data with captions.
- models_wo_captions: results of models on data without captions.
- presentation: figures in the slide deck.
- proposal: research and writeup that oriented this project.
- run_pipeline.py: program that runs our models on the data.

