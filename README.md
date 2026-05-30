# 🌸 Iris Dataset Clustering with KMeans & GMM

This project applies unsupervised machine learning techniques — **K-Means Clustering** and **Gaussian Mixture Models (GMM)** — to the classic Iris dataset in order to discover natural groupings among flower species.

---

## 📁 Project Structure

```
├── Iris_dataset_clustering_with_KMeans.ipynb   # Main notebook
├── Iris.csv                                     # Dataset
├── plots/
│   ├── 01_distributions.png                     # Histograms & boxplots
│   ├── 02_barplots_by_species.png               # Feature means by species
│   ├── 03_elbow_wcss.png                        # Elbow method
│   ├── 04_silhouette_scores.png                 # Silhouette analysis
│   ├── 05_kmeans_vs_original.png                # KMeans vs ground truth
│   └── 06_gmm_vs_original.png                  # GMM vs ground truth
└── README.md
```

---

## 📊 Dataset

- **Source:** [Iris Dataset (Kaggle)](https://www.kaggle.com/datasets/uciml/iris)
- **Rows:** 150 (after removing 1 duplicate → 149)
- **Features:** 4 numeric features + 1 target label
  | Feature | Description |
  |---|---|
  | `SepalLengthCm` | Sepal length in cm |
  | `SepalWidthCm` | Sepal width in cm |
  | `PetalLengthCm` | Petal length in cm |
  | `PetalWidthCm` | Petal width in cm |
  | `Species` | Target: Iris-setosa, Iris-versicolor, Iris-virginica |

---

## 🔄 Workflow

### 1. Data Preprocessing
- Dropped the `Id` column (non-informative)
- Checked and removed duplicate rows
- Explored distributions via histograms, boxplots, and bar plots per species
- Applied **StandardScaler** for feature normalization before clustering

### 2. Finding the Optimal Number of Clusters
Two methods were used to determine the ideal `k`:

- **Elbow Method (WCSS):** Plots inertia vs. number of clusters; the "elbow" at k=3 indicates diminishing returns
- **Silhouette Score:** Measures cluster cohesion and separation; highest score at k=3 confirms the choice

### 3. K-Means Clustering
- Fitted `KMeans(n_clusters=3)` on scaled data
- Compared predicted cluster labels against true species labels using a cross-tabulation
- Visualized results via scatter plot (Petal Length vs Petal Width)

### 4. Gaussian Mixture Model (GMM)
- Fitted `GaussianMixture(n_components=3)` as an alternative probabilistic approach
- Compared GMM predictions against ground truth similarly

---

## 📈 Results

Both models correctly identified **3 natural clusters** matching the 3 Iris species:

| Model | Accuracy (approx.) |
|---|---|
| K-Means | ~89% |
| GMM | ~96% |

> GMM slightly outperforms KMeans because it uses soft/probabilistic assignments and can model elliptical clusters — better suited to the overlapping `versicolor`/`virginica` boundary.

---

## 🖼️ Key Visualizations

| Plot | Description |
|---|---|
| `01_distributions.png` | Feature distributions — histogram + boxplot for each feature |
| `02_barplots_by_species.png` | Average feature values per species |
| `03_elbow_wcss.png` | Elbow curve to choose optimal k |
| `04_silhouette_scores.png` | Silhouette score for k = 2–10 |
| `05_kmeans_vs_original.png` | KMeans predicted clusters vs. true species |
| `06_gmm_vs_original.png` | GMM predicted clusters vs. true species |

---

## 🛠️ Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
plotly
```

Install all dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn plotly
```

---

## 🚀 How to Run

**View online (no setup needed):**
👉 [Open in nbviewer](https://nbviewer.org/github/nihadqasimov2006/Data_Science_Portfolio/blob/main/Iris_dataset_clustering_with_KMeans.ipynb)

**Run locally:**
```bash
jupyter notebook Iris_dataset_clustering_with_KMeans.ipynb
```

Make sure `Iris.csv` is in the same directory as the notebook.

---

## 📚 References

- [Scikit-learn KMeans documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [Scikit-learn GaussianMixture documentation](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)
- [Iris Dataset — UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/iris)
