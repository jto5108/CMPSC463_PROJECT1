Time-Series Segment Analysis Project
Project Description

This project analyzes time-series segments (like ABP signals) using three main steps:

Clustering – Groups similar segments together.

Closest Pair Detection – Finds the most similar segments within each cluster using Dynamic Time Warping (DTW).

Kadane’s Algorithm – Identifies intervals of significant changes in each segment.

The project can help detect trends, anomalies, and significant events in time-series data.

How to Use

Download the Dataset

Go to the Kaggle dataset page and download the required file (e.g., VitalDB_AAMI_Test_Subset.mat).

Add it to the Project

Place the downloaded dataset in your project folder.

Update File Path

Open pulse_clustering_db.py and update the dataset_path variable to point to your dataset file:

install requirements.txt

pip install -r requirements.txt

Run the Program

Execute the script using Python:

python pulse_clustering_db.py


The program will perform clustering, find closest pairs, run Kadane analysis, and generate visualizations.
