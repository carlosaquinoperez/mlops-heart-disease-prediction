# MLOps Midterm Project: Cardiovascular disease prediction

This repository contains the work for the MLOps Zoomcamp midterm project. The objective is to build an end-to-end machine learning pipeline to predict cardiovascular disease based on patient health metrics.

The project follows MLOps best practices, including environment management, a structured codebase, model serving via an API, and containerization with Docker.

## 🎯 Project Goal

The main goal is to train a binary classification model that predicts the presence (1) or absence (0) of cardiovascular disease, based on the `cardio` column in the dataset.

## 📊 Dataset

* **Source:** [Cardiovascular Disease Dataset (Kaggle)](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
* **Description:** The dataset contains 70,000 records of patient data with 11 features (like age, gender, cholesterol) and 1 target variable (`cardio`).
* **Location:** The raw dataset (`cardio_train.csv`) is located in the `/data` directory.

## 📂 Project Structure

The repository is organized to separate concerns (data, exploration, production code):