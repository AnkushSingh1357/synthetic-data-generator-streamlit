# synthetic-data-generator-streamlit

Synthetic Data Generation & Analysis Dashboard 🤖
An interactive web application built with Streamlit to generate, visualize, and evaluate synthetic tabular data using CTGAN. This tool is designed to help bridge the data gap in privacy-sensitive or data-scarce environments.

📖 About The Project
In the world of data science, high-quality data is essential, but it's often difficult to obtain due to privacy regulations (like GDPR), scarcity, or inherent biases. This project provides a practical solution by enabling users to generate high-fidelity synthetic data.

This application leverages the CTGAN (Conditional Tabular Generative Adversarial Network) model from the Synthetic Data Vault (SDV) library to learn the statistical patterns of a real dataset and then generate new, artificial data that mimics these patterns without containing any real, sensitive information. The dashboard provides an end-to-end workflow to not only generate the data but also to rigorously evaluate its quality through visual comparisons and machine learning model performance.

✨ Key Features
⬆️ CSV Upload: Easily upload your own tabular dataset to get started.

⚙️ Customizable Generation:

Manually set the number of synthetic rows to generate.

Adjust the number of Training Epochs to balance between training time and data quality.

📊 Data Analysis & Preview:

Automatically detects and lists numeric and categorical columns.

Displays a preview of both the real and the newly generated synthetic data.

📈 Distribution Comparison:

Generates overlapping density plots to visually compare the statistical distributions of numeric columns between the real and synthetic datasets.

🤖 Machine Learning Efficacy Test:

Trains a Decision Tree Classifier on both the real and synthetic datasets.

Evaluates both models on a real test set and presents a bar chart comparing their performance across Accuracy, Precision, Recall, and F1-Score.

⬇️ Download Data: Download the generated synthetic dataset as a CSV file with a single click.

🛠️ Technology Stack
Python: The core programming language.

Streamlit: For building the interactive web application.

Pandas: For data manipulation and analysis.

Scikit-learn: For machine learning model training and evaluation.

SDV (Synthetic Data Vault): For the core CTGAN synthesizer model.

Matplotlib & Seaborn: For data visualization and plotting.

