# CMOR 438 – Machine Learning (Spring 2025)

This repository contains all lecture notebooks, example code, and data for CMOR 438: Machine Learning, Spring 2025. Each lecture covers a core algorithm (from Perceptron through Reinforcement Learning), with hands-on Jupyter notebooks, visualizations, and example datasets.

---

## Repository Structure

├── data/
│ ├── palmer_penguins.csv
│ └── … # any other raw datasets
│
├── notebooks/
│ ├── Perceptron.ipynb
│ ├── LinearRegression.ipynb
│ ├── LogisticRegression.ipynb
│ ├── DeepNeuralNetwork.ipynb
│ ├── KNN.ipynb
│ ├── KNN Regression.ipynb
│ ├── DecisionTreeClassification.ipynb
│ ├── DecisionTreeRegression.ipynb
│ ├── RandomForest.ipynb
│ ├── Ensemble_Boosting_Classification.ipynb
│ ├── Ensemble_Boosting_Regression.ipynb
│ ├── KMeans_Wine_Clustering.ipynb
│ ├── PCA_BreastCancer_Analysis.ipynb
│ ├── RL_Taxi_QLearning.ipynb
│ ├── LabelPropagation_Algorithm.ipynb
│ └── MaxClique_IP.ipynb
│
├── src/
│ ├── single_neuron.py
│ └── … # any custom Python modules
│
├── requirements.txt
└── README.md


---

## Lecture-by-Lecture Contents

### Lecture 3 – The Perceptron
- **Perceptron.ipynb**  
  Single-layer binary classifier on the Palmer Penguins dataset; weight updates, decision boundary, convergence diagnostics.  
- **src/single_neuron.py**

### Lecture 4 – Linear Regression
- **LinearRegression.ipynb**  
  Gradient-descent linear regression on a real dataset with loss curves and learning-rate experiments.

### Lecture 5 – Logistic Regression
- **LogisticRegression.ipynb**  
  Binary and multiclass logistic regression, cross-entropy loss, gradient derivations, classification metrics.

### Lecture 6 – Deep Neural Networks
- **DeepNeuralNetwork.ipynb**  
  Multi-layer perceptrons: forward/backward pass, training on a small dataset, and evaluation.

### Lecture 7 – k-Nearest Neighbors
- **KNN.ipynb**  
  kNN classification: decision-boundary visualization, error vs k, classification report.  
- **KNN Regression.ipynb**  
  kNN regression on housing prices: residual analysis and hyperparameter tuning.

### Lecture 8 – Decision Trees
- **DecisionTreeClassification.ipynb**  
  Classification tree on the Breast Cancer dataset: impurity measures, pruning, model evaluation.  
- **DecisionTreeRegression.ipynb**  
  Regression tree on California Housing: MSE, MAE, R², depth experiments, feature importance.

### Lecture 9 – Random Forests & Boosting
- **RandomForest.ipynb**  
  Random Forest classification vs. bagging and stumps; feature importances; OOB score.  
- **Ensemble_Boosting_Classification.ipynb**  
  AdaBoost classification: weak learners, learning-rate curves, performance.  
- **Ensemble_Boosting_Regression.ipynb**  
  Gradient Boosting regression: validation curves, hyperparameter optimization, predicted-vs-actual plots.

### Lecture 10 – Unsupervised Learning
- **KMeans_Wine_Clustering.ipynb**  
  K-Means on the Wine dataset: elbow method, silhouette analysis, PCA visualization.  
- **PCA_BreastCancer_Analysis.ipynb**  
  PCA on the Breast Cancer dataset: explained variance, biplot, reconstruction error.

### Lecture 11 – Reinforcement Learning
- **RL_Taxi_QLearning.ipynb**  
  Q-Learning on OpenAI Gym’s Taxi-v3: MDP formulation, ε-greedy policy, learning curves, evaluation.

### Additional Graph Algorithms
- **LabelPropagation_Algorithm.ipynb**  
  Label Propagation for community detection: update rule, convergence, visualizations.  
- **MaxClique_IP.ipynb**  
  Maximum Clique via integer programming (PuLP): formulation, examples on Karate Club and random graphs.

---

## Getting Started

1. **Clone the repo**  
   ```bash
   git clone https://github.com/t0dd26/CMOR438-Spring-2025.git
   cd CMOR438-Spring-2025

2. **Install Dependencies**
    pip install -r requirements.txt
    # or with conda
    conda env create -f environment.yml

3. **Launch Jupyter**
    jupyter lab
    
Then open any notebook under notebooks/.