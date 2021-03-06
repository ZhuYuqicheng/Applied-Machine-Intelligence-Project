# TUM AMI Project: Electricity Price Forecasting

A practical Machine Learning lecture held by Professor **Klaus Diepold** at Technical University Munich (TUM). This lecture consists of reading assignment, essay writing, discussion session and final project. This repository contains codes, jupyter notebooks, report, etc. for the final project - "*Electricity Price Forecasting*".

> The spot price data obtained from MONTAL has been removed according to the regulation.

## Team Member:

- **Yuqicheng Zhu** (Responsible for: Project Management, ARIMA)
- **Runyao Yu** (Responsible for: ARIMA, Video)
- **Xuyang Zhong** (Responsible for: Transformer)
- **Han Liang** (Responsible for: Web, Data Preprocessing)
- **Junpeng Chen** (Responsible for: Web, Data Preprocessing)
- **Jiaxin Yang** (Responsible for: Report)
- **Yicong Li** (Responsible for: Report)

> Concrete contributions see: [Declaration of Contributions](./Report/Declaration_of_Contributions.md)

---

## Introduction

This project aims to compare the performance of traditional machine learning and deep learning in predicting electricity prices. We selected **ARIMA** and **Transformer** as the representor for traditional training and deep learning respectively. They were compared in terms of accuracy, stability, efficiency and interpretability, etc. All processes and corresponding findings and results were documented in jupyter notebooks. A summary of this project can be found in report.

## Poster

![Video Preview](./Report/poster.png)

## Content List

- [Website Guideline](./Web/web/guide.md)
- [ARIMA Code+Documentation](#arima)
- [Transformer Code+Documentation](#transformer)
- [Report](#report)
- [Video](#video)

---
## ARIMA

All processing, tests and experiments of ARIMA were carried out in Jyupter Notebook. For more details, you can access our notebooks with following links or a [summarized documentation](./ARIMA/ARIMA_Documentation.md).

- [Missing Value Processing](./ARIMA/0_label_missing_processing.ipynb)
- [Feature Extraction](./ARIMA/1_feature_extraction.ipynb) (That might take very long time!)
- [Feature Selection](./ARIMA/2_feature_selection.ipynb)
- [Training Length Experiments](./ARIMA/3_training_length.ipynb)
- [Feature Reduction Experiments](./ARIMA/4_feature_number.ipynb)
- [Simple Model Experiments](./ARIMA/ARIMA_experiments/dummy_experimens_simple_models)
- [PCA Experiments](./ARIMA/ARIMA_experiments/pca_plus_original_features_experiment)
- [Exogenous Feature Experiments](./ARIMA/ARIMA_experiments/with_or_without_exogenous_features_experiement)
- [Online Training Strategy](./ARIMA/5_Online_training.ipynb)
- [Pre-trained Strategy](./ARIMA/6_Pre_trained_solution.ipynb)
- [Model Deployment](./ARIMA/7_model_deployment.ipynb)

---

## Transformer

> The Code is mainly based on this repository: https://github.com/nklingen/Transformer-Time-Series-Forecasting

For more information about transformer implementation, check out [this documentation](./Transformer/README.md).

- Requirements
- Data Information
- Preprocessing
- Model Structure
- Model Training
- Inference
- Discussion

> To test the model just run the [main script](./Transformer/main.py)

---

## Report

---

You can find our final version report [here](./Report)

---
 
## Video

![Video Preview](./Video/GitLab_Video_Preview.gif)

Check out our project video in YouTube: https://www.youtube.com/watch?v=_SZ6bKFkFyQ
