#library........

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

#loaddataset......

df = pd.read_csv('/Users/aakashkumar/Desktop/election.csv', encoding='ISO-8859-1')


print("Shape:", df.shape)
print(df.head(3))
print("\nDtypes:\n", df.dtypes)

# Basic cleaning
df.columns = [c.strip() for c in df.columns]
df = df.drop_duplicates()

# Ensure numeric columns are numeric
for col in ["EVM_votes", "postal_votes", "total_votes", "rank", "election_year"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Create a common target for classification: winner vs not winner
df["is_winner"] = (df["rank"] == 1).astype(int)

# Quick missing values report
print("\nMissing values:\n", df.isna().sum().sort_values(ascending=False).head(10))



#Introduction + Data Preprocessing (encoding, scaling, splitting, pipelines)
# UNIT I: DATA PREPROCESSING


from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Choose features (avoid leakage depending on task)
base_features = [
    "state_name", "constituency_name", "constituency_type",
    "party_name", "candidate_name",
    "EVM_votes", "postal_votes"
]

# Keep only available columns
base_features = [c for c in base_features if c in df.columns]
data = df[base_features + ["total_votes", "is_winner"]].copy()

# Separate feature types
numeric_features = [c for c in base_features if data[c].dtype != "object"]
categorical_features = [c for c in base_features if data[c].dtype == "object"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="drop"
)

print("Numeric:", numeric_features)
print("Categorical:", categorical_features)

# Example split for REGRESSION target: total_votes
X = data[base_features]
y_reg = data["total_votes"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

print("Train:", X_train.shape, "Test:", X_test.shape)




#Supervised Learning – Regression (SLR, MLR, Poly, Logistic, OLS, Correlation, Metrics)
#OLS + Correlations (statsmodels)
# UNIT II-A: OLS + CORRELATION

import statsmodels.api as sm

# Simple correlation on numeric columns
num_cols = [c for c in ["EVM_votes","postal_votes","total_votes","rank"] if c in df.columns]
print(df[num_cols].corr(numeric_only=True))

# OLS example: total_votes ~ EVM_votes + postal_votes (no categoricals)
tmp = df[["total_votes","EVM_votes","postal_votes"]].dropna().copy()
X_ols = tmp[["EVM_votes","postal_votes"]]
X_ols = sm.add_constant(X_ols)
y_ols = tmp["total_votes"]

ols_model = sm.OLS(y_ols, X_ols).fit()
print(ols_model.summary())






#Regression Models + Evaluation (MAE, MSE, RMSE, R²)
# UNIT II-B: REGRESSION MODELS + METRICS


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1) Multiple Linear Regression (with preprocessing for categoricals)
mlr = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", LinearRegression())
])

mlr.fit(X_train, y_train)
pred = mlr.predict(X_test)

mae = mean_absolute_error(y_test, pred)
mse = mean_squared_error(y_test, pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, pred)

print("MLR -> MAE:", mae, "MSE:", mse, "RMSE:", rmse, "R2:", r2)

# 2) Polynomial Regression (numeric-only example to keep it clean)
from sklearn.pipeline import Pipeline

poly_data = df[["EVM_votes","postal_votes","total_votes"]].dropna().copy()
Xp = poly_data[["EVM_votes","postal_votes"]]
yp = poly_data["total_votes"]

Xp_train, Xp_test, yp_train, yp_test = train_test_split(Xp, yp, test_size=0.2, random_state=42)

poly_reg = Pipeline(steps=[
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("lin", LinearRegression())
])

poly_reg.fit(Xp_train, yp_train)
pp = poly_reg.predict(Xp_test)

print("Poly(deg=2) -> RMSE:", np.sqrt(mean_squared_error(yp_test, pp)), "R2:", r2_score(yp_test, pp))





#Logistic Regression (classification but included in Unit II list)
               #LOGISTIC REGRESSION (is_winner)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

y_clf = data["is_winner"]
X = data[base_features]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

logreg = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", LogisticRegression(max_iter=2000))
])

logreg.fit(X_train, y_train)
pred = logreg.predict(X_test)

print("LogReg Accuracy:", accuracy_score(y_test, pred))




#Supervised Learning – Classification (KNN, Naive Bayes, DT, SVM + Metrics)
#CLASSIFICATION MODELS + METRICS

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, log_loss
)

# We'll use is_winner as classification target
y = data["is_winner"]
X = data[base_features]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# NOTE: MultinomialNB expects non-negative features; easiest is to use one-hot only + no scaling.
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

cat_only = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)],
    remainder="drop"
)

models = {}

# KNN
models["KNN(k=7)"] = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", KNeighborsClassifier(n_neighbors=7))
])

# Decision Tree
models["DecisionTree"] = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", DecisionTreeClassifier(random_state=42, max_depth=8))
])

# Linear SVM
models["LinearSVM"] = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", LinearSVC())
])

# Naive Bayes (categorical-only onehot)
models["MultinomialNB(cat-only)"] = Pipeline(steps=[
    ("preprocess", cat_only),
    ("model", MultinomialNB(alpha=1.0))
])

def eval_classifier(name, model):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    cm = confusion_matrix(y_test, pred)

    print(f"\n=== {name} ===")
    print("Accuracy:", acc)
    print("Precision:", prec, "Recall:", rec, "F1:", f1)
    print("Confusion Matrix:\n", cm)

    # AUC + LogLoss require probabilities or decision scores
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
        print("LogLoss:", log_loss(y_test, model.predict_proba(X_test)))
        print("AUC:", roc_auc_score(y_test, proba))
    elif hasattr(model, "decision_function"):
        score = model.decision_function(X_test)
        # For AUC, decision scores are okay
        print("AUC:", roc_auc_score(y_test, score))

for name, m in models.items():
    eval_classifier(name, m)










#Unsupervised Learning – KMeans, Hierarchical, Linkages, Association Rules, Market Basket
    #FEATURE ENGINEERING FOR CLUSTERING
# Aggregate at constituency level (each constituency has multiple candidates)



grp_cols = ["state_name", "constituency_name"]
agg = df.dropna(subset=grp_cols + ["total_votes"]).groupby(grp_cols).agg(
    n_candidates=("candidate_name", "count"),
    total_votes_sum=("total_votes", "sum"),
    top_votes=("total_votes", "max"),
    avg_votes=("total_votes", "mean"),
    postal_sum=("postal_votes", "sum"),
    evm_sum=("EVM_votes", "sum"),
).reset_index()

agg["top_vote_share"] = agg["top_votes"] / agg["total_votes_sum"].replace(0, np.nan)
agg = agg.dropna(subset=["top_vote_share"]).copy()

print(agg.head(3))




#B) KMeans (+ choosing K, init trap note)

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

cluster_features = ["n_candidates","total_votes_sum","avg_votes","top_vote_share","postal_sum","evm_sum"]
Xc = agg[cluster_features].copy()

scaler = StandardScaler()
Xc_scaled = scaler.fit_transform(Xc)

# Elbow method (SSE / inertia) to help pick K
inertias = {}
for k in range(2, 11):
    km = KMeans(n_clusters=k, n_init=20, random_state=42)  # n_init helps reduce random init trap
    km.fit(Xc_scaled)
    inertias[k] = km.inertia_

print("Inertias:", inertias)

# Train final KMeans (example k=5)
k = 5
kmeans = KMeans(n_clusters=k, n_init=30, random_state=42)
agg["kmeans_cluster"] = kmeans.fit_predict(Xc_scaled)

print(agg[["state_name","constituency_name","kmeans_cluster"]].head(10))
print("\nCluster counts:\n", agg["kmeans_cluster"].value_counts())


#Hierarchical clustering (Agglomerative + linkage styles)

from sklearn.cluster import AgglomerativeClustering

# linkage options in sklearn: "ward" (requires euclidean), "complete", "average", "single"
for linkage in ["ward", "complete", "average", "single"]:
    model = AgglomerativeClustering(n_clusters=5, linkage=linkage)
    labels = model.fit_predict(Xc_scaled)
    print(linkage, "clusters:", np.bincount(labels))





#Association Rules (Market Basket) – “Which parties co-occur in constituencies?


from collections import Counter
from itertools import combinations

# Basket = parties present in each constituency
basket_df = df.dropna(subset=["state_name","constituency_name","party_name"]).copy()
baskets = basket_df.groupby(["state_name","constituency_name"])["party_name"].apply(
    lambda s: sorted(set(s))
).tolist()

N = len(baskets)
item_counts = Counter()
pair_counts = Counter()

for items in baskets:
    for it in items:
        item_counts[it] += 1
    for a,b in combinations(items, 2):
        pair_counts[(a,b)] += 1

def support(x): 
    return x / N

# Build rules A -> B using pairs
rules = []
for (a,b), ab_cnt in pair_counts.items():
    sup_ab = support(ab_cnt)
    sup_a = support(item_counts[a])
    sup_b = support(item_counts[b])

    # A -> B
    conf_a_b = sup_ab / sup_a if sup_a else 0
    lift_a_b = conf_a_b / sup_b if sup_b else 0
    rules.append((a, b, sup_ab, conf_a_b, lift_a_b))

    # B -> A
    conf_b_a = sup_ab / sup_b if sup_b else 0
    lift_b_a = conf_b_a / sup_a if sup_a else 0
    rules.append((b, a, sup_ab, conf_b_a, lift_b_a))

rules_df = pd.DataFrame(rules, columns=["A","B","support","confidence","lift"])

# Filter for meaningful rules
strong = rules_df[(rules_df["support"] >= 0.10) & (rules_df["confidence"] >= 0.60)].sort_values(
    ["lift","confidence","support"], ascending=False
)

print("Total baskets:", N)
print("\nTop strong rules:\n", strong.head(20))








#Dimensionality Reduction + Neural Networks (PCA, MLP, CNN/RNN examples)
#A) PCA (on constituency features)


from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=42)
Z = pca.fit_transform(Xc_scaled)

print("Explained variance ratio:", pca.explained_variance_ratio_)
agg["pca1"] = Z[:, 0]
agg["pca2"] = Z[:, 1]
print(agg[["pca1","pca2","kmeans_cluster"]].head(5))




#Feedforward Neural Network / MLP (sklearn)
# MLP (NEURAL NETWORK) - CLASSIFICATION

from sklearn.neural_network import MLPClassifier

mlp = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        max_iter=30,
        random_state=42
    ))
])

mlp.fit(X_train, y_train)
pred = mlp.predict(X_test)

print("MLP Accuracy:", accuracy_score(y_test, pred))
print("MLP F1:", f1_score(y_test, pred, zero_division=0))


#PyTorch MLP skeleton (for “full syllabus coverage”)
# UNIT V-C: PYTORCH MLP SKELETON (OPTIONAL)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


#Model Performance (bias-variance, LOOCV, KFold, Bagging, Boosting, Random Forest)


from sklearn.model_selection import KFold, cross_val_score, LeaveOneOut
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier

# We'll evaluate classification models with cross-validation
X = data[base_features]
y = data["is_winner"]

# 1) K-Fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)

rf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        max_depth=None,
        n_jobs=-1
    ))
])

scores = cross_val_score(rf, X, y, cv=kf, scoring="f1")
print("RandomForest 5-fold F1:", scores, "Mean:", scores.mean())

# 2) Bagging (with Decision Tree base)
bag = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", BaggingClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ))
])

scores = cross_val_score(bag, X, y, cv=kf, scoring="f1")
print("Bagging 5-fold F1 mean:", scores.mean())

# 3) Boosting
gb = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", GradientBoostingClassifier(random_state=42))
])

scores = cross_val_score(gb, X, y, cv=kf, scoring="f1")
print("GradientBoosting 5-fold F1 mean:", scores.mean())

# 4) LOOCV (expensive on big datasets) -> do on a sample
sample = data.sample(n=500, random_state=42)
X_s = sample[base_features]
y_s = sample["is_winner"]

loo = LeaveOneOut()
scores = cross_val_score(gb, X_s, y_s, cv=loo, scoring="accuracy")
print("LOOCV (sample=500) Accuracy mean:", scores.mean())




















# Data Preprocessing Graphs (Missing values, distributions, outliers)
def plot_missing_values(df, top_n=20):
    miss = df.isna().sum().sort_values(ascending=False).head(top_n)
    miss = miss[miss > 0]
    plt.figure()
    plt.bar(miss.index.astype(str), miss.values)
    plt.xticks(rotation=75, ha="right")
    plt.title(f"Top {top_n} Missing Values by Column")
    plt.ylabel("Missing Count")
    plt.tight_layout()
    plt.show()

plot_missing_values(df, top_n=20)
def plot_missing_values(df, top_n=20):
    miss = df.isna().sum().sort_values(ascending=False).head(top_n)
    miss = miss[miss > 0]
    plt.figure()
    plt.bar(miss.index.astype(str), miss.values)
    plt.xticks(rotation=75, ha="right")
    plt.title(f"Top {top_n} Missing Values by Column")
    plt.ylabel("Missing Count")
    plt.tight_layout()
    plt.show()

plot_missing_values(df, top_n=20)




#Numeric distributions (histograms)
def plot_numeric_histograms(df, cols):
    for c in cols:
        if c in df.columns:
            x = df[c].dropna()
            plt.figure()
            plt.hist(x, bins=30)
            plt.title(f"Histogram: {c}")
            plt.xlabel(c)
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.show()

plot_numeric_histograms(df, ["EVM_votes", "postal_votes", "total_votes", "rank"])




#Boxplots (outliers)
def plot_boxplots(df, cols):
    for c in cols:
        if c in df.columns:
            x = df[c].dropna()
            plt.figure()
            plt.boxplot(x, vert=True)
            plt.title(f"Boxplot: {c}")
            plt.ylabel(c)
            plt.tight_layout()
            plt.show()

plot_boxplots(df, ["EVM_votes", "postal_votes", "total_votes"])


#Category frequency bars (Top K)

def plot_top_categories(df, col, top_k=15):
    if col not in df.columns:
        return
    vc = df[col].astype(str).value_counts().head(top_k)
    plt.figure()
    plt.bar(vc.index, vc.values)
    plt.xticks(rotation=75, ha="right")
    plt.title(f"Top {top_k} Categories: {col}")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

plot_top_categories(df, "party_name", top_k=20)
plot_top_categories(df, "state_name", top_k=20)


#Regression Graphs (Correlation heatmap, actual vs predicted, residuals, polynomial fit)

def plot_corr_heatmap(df, cols):
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr(numeric_only=True)

    plt.figure()
    im = plt.imshow(corr.values, aspect="auto")
    plt.colorbar(im)
    plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
    plt.yticks(range(len(cols)), cols)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

#plot_corr_heatmap(df, ["EVM_votes", "postal_votes", "total_votes", "rank"])
#Correlation “heatmap” (Matplotlib)

def plot_corr_heatmap(df, cols):
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr(numeric_only=True)

    plt.figure()
    im = plt.imshow(corr.values, aspect="auto")
    plt.colorbar(im)
    plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
    plt.yticks(range(len(cols)), cols)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

plot_corr_heatmap(df, ["EVM_votes", "postal_votes", "total_votes", "rank"])




#Multiple Linear Regression: Actual vs Predicted + Residual plot
# Features
base_features = ["state_name","constituency_name","constituency_type",
                 "party_name","candidate_name","EVM_votes","postal_votes"]
base_features = [c for c in base_features if c in df.columns]

data = df[base_features + ["total_votes"]].dropna(subset=["total_votes"]).copy()

X = data[base_features]
y = data["total_votes"]

num_features = [c for c in base_features if data[c].dtype != "object"]
cat_features = [c for c in base_features if data[c].dtype == "object"]

preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_features),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_features)
    ],
    remainder="drop"
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlr = Pipeline([("prep", preprocess), ("model", LinearRegression())])
mlr.fit(X_train, y_train)
pred = mlr.predict(X_test)

# Actual vs Predicted
plt.figure()
plt.scatter(y_test, pred, s=12)
plt.title("Regression: Actual vs Predicted (MLR)")
plt.xlabel("Actual total_votes")
plt.ylabel("Predicted total_votes")
plt.tight_layout()
plt.show()

# Residuals
res = y_test.values - pred
plt.figure()
plt.scatter(pred, res, s=12)
plt.axhline(0)
plt.title("Regression: Residual Plot (Predicted vs Residuals)")
plt.xlabel("Predicted")
plt.ylabel("Residual (Actual - Predicted)")
plt.tight_layout()
plt.show()



#Polynomial Regression curve (numeric-only example)
poly_df = df[["EVM_votes","total_votes"]].dropna()
Xp = poly_df[["EVM_votes"]].values
yp = poly_df["total_votes"].values

Xp_train, Xp_test, yp_train, yp_test = train_test_split(Xp, yp, test_size=0.2, random_state=42)

poly_model = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("lin", LinearRegression())
])
poly_model.fit(Xp_train, yp_train)

# Smooth curve plot
x_line = np.linspace(Xp.min(), Xp.max(), 200).reshape(-1, 1)
y_line = poly_model.predict(x_line)

plt.figure()
plt.scatter(Xp_test, yp_test, s=10)
plt.plot(x_line, y_line, linewidth=2)
plt.title("Polynomial Regression (degree=2): total_votes vs EVM_votes")
plt.xlabel("EVM_votes")
plt.ylabel("total_votes")
plt.tight_layout()
plt.show()






#Classification Graphs (Confusion matrix, ROC, PR curve, probability hist)
data_c = df[base_features + ["is_winner"]].dropna(subset=["is_winner"]).copy()
X = data_c[base_features]
y = data_c["is_winner"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

logreg = Pipeline([("prep", preprocess), ("model", LogisticRegression(max_iter=3000))])
logreg.fit(X_train, y_train)

proba = logreg.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)


#Confusion matrix plot
def plot_confusion_matrix(cm, class_names=("0","1"), title="Confusion Matrix"):
    plt.figure()
    plt.imshow(cm, aspect="auto")
    plt.colorbar()
    plt.xticks([0,1], class_names)
    plt.yticks([0,1], class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)

    # annotate
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.show()

cm = confusion_matrix(y_test, pred)
plot_confusion_matrix(cm, class_names=("Not Winner","Winner"), title="LogReg Confusion Matrix")




#ROC Curve + AUC
fpr, tpr, _ = roc_curve(y_test, proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.4f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.title("ROC Curve (Logistic Regression)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.show()


#Precision–Recall curve
prec, rec, _ = precision_recall_curve(y_test, proba)
ap = average_precision_score(y_test, proba)

plt.figure()
plt.plot(rec, prec, linewidth=2, label=f"AP = {ap:.4f}")
plt.title("Precision-Recall Curve (Logistic Regression)")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.tight_layout()
plt.show()



#Predicted probability distribution
plt.figure()
plt.hist(proba[y_test.values == 0], bins=30, alpha=0.7, label="Actual: Not Winner")
plt.hist(proba[y_test.values == 1], bins=30, alpha=0.7, label="Actual: Winner")
plt.title("Predicted Probability Distribution")
plt.xlabel("Predicted P(Winner)")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.show()


#Clustering Graphs (Elbow, PCA scatter, cluster sizes, Dendrogram)
grp_cols = ["state_name", "constituency_name"]
agg = df.dropna(subset=grp_cols + ["total_votes"]).groupby(grp_cols).agg(
    n_candidates=("candidate_name", "count"),
    total_votes_sum=("total_votes", "sum"),
    top_votes=("total_votes", "max"),
    avg_votes=("total_votes", "mean"),
    postal_sum=("postal_votes", "sum"),
    evm_sum=("EVM_votes", "sum"),
).reset_index()

agg["top_vote_share"] = agg["top_votes"] / agg["total_votes_sum"].replace(0, np.nan)
agg = agg.dropna(subset=["top_vote_share"]).copy()

cluster_features = ["n_candidates","total_votes_sum","avg_votes","top_vote_share","postal_sum","evm_sum"]
Xc = agg[cluster_features].values

scaler = StandardScaler()
Xc_scaled = scaler.fit_transform(Xc)


#Elbow curve (K selection)
inertias = []
ks = list(range(2, 11))

for k in ks:
    km = KMeans(n_clusters=k, n_init=30, random_state=42)
    km.fit(Xc_scaled)
    inertias.append(km.inertia_)

plt.figure()
plt.plot(ks, inertias, marker="o")
plt.title("Elbow Method for KMeans")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia (SSE)")
plt.tight_layout()
plt.show()



#Clustered PCA scatter (2D view)

k = 5
kmeans = KMeans(n_clusters=k, n_init=30, random_state=42)
labels = kmeans.fit_predict(Xc_scaled)

pca = PCA(n_components=2, random_state=42)
Z = pca.fit_transform(Xc_scaled)

plt.figure()
plt.scatter(Z[:,0], Z[:,1], c=labels, s=12)
plt.title("KMeans Clusters visualized in PCA (2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()

# cluster sizes bar
counts = pd.Series(labels).value_counts().sort_index()
plt.figure()
plt.bar(counts.index.astype(str), counts.values)
plt.title("KMeans Cluster Sizes")
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.tight_layout()
plt.show()




#Hierarchical clustering dendrogram (if SciPy installed)
if SCIPY_OK:
    # Use a sample for readability
    sample_n = min(400, Xc_scaled.shape[0])
    Xs = Xc_scaled[:sample_n]

    Zlink = linkage(Xs, method="ward")  # ward/complete/average/single
    plt.figure(figsize=(12, 6))
    dendrogram(Zlink, truncate_mode="lastp", p=30)
    plt.title("Hierarchical Clustering Dendrogram (truncated)")
    plt.xlabel("Cluster")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.show()
else:
    print("SciPy not available: install scipy to plot dendrogram.")



#Association Rules Graphs (Top rules bar charts)
from collections import Counter
from itertools import combinations

basket_df = df.dropna(subset=["state_name","constituency_name","party_name"]).copy()
baskets = basket_df.groupby(["state_name","constituency_name"])["party_name"].apply(
    lambda s: sorted(set(s))
).tolist()

N = len(baskets)
item_counts = Counter()
pair_counts = Counter()

for items in baskets:
    for it in items:
        item_counts[it] += 1
    for a,b in combinations(items, 2):
        pair_counts[(a,b)] += 1

def support(x): 
    return x / N

rules = []
for (a,b), ab_cnt in pair_counts.items():
    sup_ab = support(ab_cnt)
    sup_a = support(item_counts[a])
    sup_b = support(item_counts[b])

    conf_a_b = sup_ab / sup_a if sup_a else 0
    lift_a_b = conf_a_b / sup_b if sup_b else 0
    rules.append((f"{a} -> {b}", sup_ab, conf_a_b, lift_a_b))

rules_df = pd.DataFrame(rules, columns=["rule","support","confidence","lift"])
top_lift = rules_df.sort_values("lift", ascending=False).head(15)
top_conf = rules_df.sort_values("confidence", ascending=False).head(15)

def plot_rule_bars(df_rules, value_col, title):
    plt.figure()
    plt.barh(df_rules["rule"][::-1], df_rules[value_col][::-1])
    plt.title(title)
    plt.xlabel(value_col)
    plt.tight_layout()
    plt.show()

plot_rule_bars(top_lift, "lift", "Top 15 Association Rules by Lift")
plot_rule_bars(top_conf, "confidence", "Top 15 Association Rules by Confidence")



#PCA Graphs (Explained variance, scree plot)
pca_full = PCA(n_components=min(10, Xc_scaled.shape[1]), random_state=42)
pca_full.fit(Xc_scaled)

evr = pca_full.explained_variance_ratio_
cum = np.cumsum(evr)

plt.figure()
plt.plot(range(1, len(evr)+1), evr, marker="o")
plt.title("PCA Scree Plot (Explained Variance Ratio)")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(range(1, len(cum)+1), cum, marker="o")
plt.title("PCA Cumulative Explained Variance")
plt.xlabel("Principal Component")
plt.ylabel("Cumulative Explained Variance")
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()



#Model Performance Graphs (Learning curve, CV scores comparison)
def plot_learning_curve(estimator, X, y, title="Learning Curve"):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=5, scoring="f1", n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 6), shuffle=True, random_state=42
    )

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    plt.figure()
    plt.plot(train_sizes, train_mean, marker="o", label="Train F1")
    plt.plot(train_sizes, val_mean, marker="o", label="CV F1")
    plt.title(title)
    plt.xlabel("Training Samples")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example: Random Forest
rf = Pipeline([("prep", preprocess), ("model", RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1))])
plot_learning_curve(rf, data_c[base_features], data_c["is_winner"], title="Learning Curve: RandomForest (F1)")



#Compare models via CV (bar chart)
from sklearn.model_selection import cross_val_score

models = {
    "LogReg": Pipeline([("prep", preprocess), ("model", LogisticRegression(max_iter=3000))]),
    "KNN": Pipeline([("prep", preprocess), ("model", KNeighborsClassifier(n_neighbors=7))]),
    "DecisionTree": Pipeline([("prep", preprocess), ("model", DecisionTreeClassifier(random_state=42, max_depth=8))]),
    "RandomForest": Pipeline([("prep", preprocess), ("model", RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1))]),
    "GradientBoost": Pipeline([("prep", preprocess), ("model", GradientBoostingClassifier(random_state=42))]),
}

X_all = data_c[base_features]
y_all = data_c["is_winner"].astype(int)

names = []
means = []

for name, est in models.items():
    scores = cross_val_score(est, X_all, y_all, cv=5, scoring="f1")
    names.append(name)
    means.append(scores.mean())

plt.figure()
plt.bar(names, means)
plt.title("Model Comparison (5-fold CV Mean F1)")
plt.xlabel("Model")
plt.ylabel("Mean F1")
plt.tight_layout()
plt.show()


























































































































































