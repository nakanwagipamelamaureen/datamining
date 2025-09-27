
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.utils import shuffle
import time
import warnings
warnings.filterwarnings("ignore")

digits = load_digits()
X_img = digits.images  # shape (n_samples, 8, 8)
y = digits.target
print("Using sklearn digits dataset, shape:", X_img.shape, "labels:", np.unique(y))
# Flatten and scale to 0-1 (pixel max is 16 for load_digits)
X_flat = X_img.reshape((X_img.shape[0], -1)).astype(np.float32) / 16.0

# Basic splits
X_train_val, X_hold, y_train_val, y_hold = train_test_split(X_flat, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val)

print("Train/Val/Hold shapes:", X_train.shape, X_val.shape, X_hold.shape)

# --- EDA ---
rng = np.random.RandomState(0)
sample_idx = rng.choice(np.arange(X_img.shape[0]), size=16, replace=False)

fig, axes = plt.subplots(4,4, figsize=(6,6))
fig.suptitle("Random sample digit images (8x8 dataset)")
for ax, idx in zip(axes.flatten(), sample_idx):
    ax.imshow(X_img[idx], cmap='gray', interpolation='nearest')
    ax.set_title(f"{y[idx]}")
    ax.axis('off')
plt.tight_layout()
plt.show()

# Histogram
plt.figure(figsize=(7,3))
plt.hist(y, bins=np.arange(11)-0.5)
plt.title("Label distribution (digits dataset)")
plt.xlabel("Digit")
plt.ylabel("Count")
plt.xticks(range(10))
plt.show()

# Average images by class
plt.figure(figsize=(8,3))
for digit in range(10):
    avg = X_img[y==digit].mean(axis=0)
    plt.subplot(2,5,digit+1)
    plt.imshow(avg, cmap='gray', interpolation='nearest')
    plt.title(f"avg {digit}")
    plt.axis('off')
plt.suptitle("Average images per class (digits dataset)")
plt.tight_layout()
plt.show()

# PCA scatter (subset)
subset_size = min(1000, X_flat.shape[0])
Xsub, ysub = shuffle(X_flat, y, random_state=42)
Xsub = Xsub[:subset_size]; ysub = ysub[:subset_size]
pca2 = PCA(n_components=2, random_state=42)
X_pca2 = pca2.fit_transform(Xsub)
plt.figure(figsize=(7,5))
plt.scatter(X_pca2[:,0], X_pca2[:,1], c=ysub, s=12)
plt.title("PCA(2) scatter of digits subset")
plt.colorbar(boundaries=np.arange(11)-0.5, ticks=np.arange(10))
plt.show()

# --- Feature engineering ---
def symmetry_score(img_flat, size=8):
    img = img_flat.reshape(size,size)
    flipped = np.fliplr(img)
    return np.mean(np.abs(img - flipped))

def pixel_density(img_flat):
    return np.mean(img_flat > 0.0)

def center_of_mass_features(img_flat, size=8):
    img = img_flat.reshape(size,size)
    total = np.sum(img)
    if total == 0:
        return (0.0, 0.0)
    coords = np.indices(img.shape)
    cx = np.sum(coords[1] * img) / total
    cy = np.sum(coords[0] * img) / total
    return (cx/(size-1), cy/(size-1))

def compute_features(X_flat_in):
    sym = np.array([symmetry_score(x, size=8) for x in X_flat_in])
    dens = np.array([pixel_density(x) for x in X_flat_in])
    com = np.array([center_of_mass_features(x, size=8) for x in X_flat_in])
    comx = com[:,0]; comy = com[:,1]
    feats = np.vstack([sym, dens, comx, comy]).T
    return feats

feats_train = compute_features(X_train)
feats_val = compute_features(X_val)
feats_hold = compute_features(X_hold)

feats_df = pd.DataFrame(feats_train, columns=["symmetry", "density", "com_x", "com_y"])
print("Engineered features sample stats:\n", feats_df.describe().T)

# PCA50 adapted to smaller data: we'll use PCA with n_components=20
pca20 = PCA(n_components=20, random_state=42)
pca20.fit(X_train)
Xtrain_pca20 = pca20.transform(X_train)
Xval_pca20 = pca20.transform(X_val)
Xhold_pca20 = pca20.transform(X_hold)

Xtrain_fe = np.hstack([Xtrain_pca20, feats_train])
Xval_fe = np.hstack([Xval_pca20, feats_val])
Xhold_fe = np.hstack([Xhold_pca20, feats_hold])

# t-SNE visualization on subset
tsne_subset = min(800, Xtrain_fe.shape[0])
Xtsne_sub, ytsne_sub = shuffle(Xtrain_fe, y_train, random_state=0)
Xtsne_sub = Xtsne_sub[:tsne_subset]; ytsne_sub = ytsne_sub[:tsne_subset]
tsne = TSNE(n_components=2, perplexity=30, n_iter=600, random_state=42)
X_tsne_2 = tsne.fit_transform(Xtsne_sub)
plt.figure(figsize=(7,5))
plt.scatter(X_tsne_2[:,0], X_tsne_2[:,1], c=ytsne_sub, s=8)
plt.title("t-SNE on PCA20 + engineered features (subset)")
plt.colorbar(boundaries=np.arange(11)-0.5, ticks=np.arange(10))
plt.show()

# --- Compare k-NN on raw vs reduced ---
knn_raw = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
knn_reduced = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)

t0 = time.time()
knn_raw.fit(X_train, y_train)
t_train_knn_raw = time.time() - t0
t0 = time.time()
pred_knn_raw = knn_raw.predict(X_val)
t_pred_knn_raw = time.time() - t0
acc_knn_raw = accuracy_score(y_val, pred_knn_raw)
print(f"k-NN raw acc: {acc_knn_raw:.4f}, train {t_train_knn_raw:.3f}s, pred {t_pred_knn_raw:.3f}s")

t0 = time.time()
knn_reduced.fit(Xtrain_fe, y_train)
t_train_knn_red = time.time() - t0
t0 = time.time()
pred_knn_red = knn_reduced.predict(Xval_fe)
t_pred_knn_red = time.time() - t0
acc_knn_red = accuracy_score(y_val, pred_knn_red)
print(f"k-NN reduced acc: {acc_knn_red:.4f}, train {t_train_knn_red:.3f}s, pred {t_pred_knn_red:.3f}s")

# --- Preprocessing: standardize engineered+PCA features ---
scaler = StandardScaler()
Xtrain_fe_scaled = scaler.fit_transform(Xtrain_fe)
Xval_fe_scaled = scaler.transform(Xval_fe)
Xhold_fe_scaled = scaler.transform(Xhold_fe)

# --- Train models ---
mlp_flat = MLPClassifier(hidden_layer_sizes=(50,), max_iter=300, random_state=42)
mlp_fe = MLPClassifier(hidden_layer_sizes=(50,), max_iter=300, random_state=42)
bag_dt_flat = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=15, random_state=42, n_jobs=-1)
rf_flat = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

models = {
    "knn_raw": knn_raw,
    "knn_reduced": knn_reduced,
    "mlp_flat": mlp_flat,
    "mlp_fe": mlp_fe,
    "bag_dt_flat": bag_dt_flat,
    "rf_flat": rf_flat
}

results = []
for name, model in models.items():
    print("\nTraining:", name)
    if name.endswith("_flat") or name=="knn_raw" or name=="bag_dt_flat" or name=="rf_flat":
        Xfit = X_train
        Xval_in = X_val
        Xhold_in = X_hold
    else:
        Xfit = Xtrain_fe_scaled if name.endswith("_fe") or name=="knn_reduced" else X_train
        Xval_in = Xval_fe_scaled if name.endswith("_fe") or name=="knn_reduced" else X_val
        Xhold_in = Xhold_fe_scaled if name.endswith("_fe") or name=="knn_reduced" else X_hold

    if name == "knn_reduced":
        t0 = time.time()
        pred = model.predict(Xhold_in)
        pred_time = time.time() - t0
        train_time = 0.0
    else:
        t0 = time.time()
        model.fit(Xfit, y_train)
        train_time = time.time() - t0
        t0 = time.time()
        pred = model.predict(Xhold_in)
        pred_time = time.time() - t0

    acc = accuracy_score(y_hold, pred)
    prec = precision_score(y_hold, pred, average='macro', zero_division=0)
    rec = recall_score(y_hold, pred, average='macro', zero_division=0)
    results.append({
        "model": name,
        "accuracy": acc,
        "precision_macro": prec,
        "recall_macro": rec,
        "train_time_s": train_time,
        "pred_time_s": pred_time
    })
    print(f"{name} -> acc {acc:.4f}, prec {prec:.4f}, rec {rec:.4f}, train {train_time:.3f}s, pred {pred_time:.3f}s")

results_df = pd.DataFrame(results).set_index("model").sort_values("accuracy", ascending=False)
import caas_jupyter_tools as cjt
cjt.display_dataframe_to_user("Model comparison on holdout set (digits dataset)", results_df)

# --- Robustness: add Gaussian noise and evaluate ---
def add_gaussian_noise(X_flat_in, std=0.2):
    noisy = X_flat_in + np.random.normal(loc=0.0, scale=std, size=X_flat_in.shape)
    noisy = np.clip(noisy, 0.0, 1.0)
    return noisy

noise_levels = [0.1, 0.2, 0.4]
robustness_rows = []
for std in noise_levels:
    X_hold_noisy = add_gaussian_noise(X_hold, std=std)
    feats_hold_noisy = compute_features(X_hold_noisy)
    Xhold_pca20_noisy = pca20.transform(X_hold_noisy)
    Xhold_fe_noisy = np.hstack([Xhold_pca20_noisy, feats_hold_noisy])
    Xhold_fe_noisy_scaled = scaler.transform(Xhold_fe_noisy)

    for row in results_df.reset_index().itertuples():
        name = row.model
        model = models[name]
        if name.endswith("_flat") or name=="knn_raw" or name=="bag_dt_flat" or name=="rf_flat":
            X_in = X_hold_noisy
        else:
            X_in = Xhold_fe_noisy_scaled if name.endswith("_fe") or name=="knn_reduced" else X_hold_noisy
        try:
            t0 = time.time()
            pred_noisy = model.predict(X_in)
            pred_time = time.time() - t0
            acc_noisy = accuracy_score(y_hold, pred_noisy)
            robustness_rows.append({
                "model": name,
                "noise_std": std,
                "accuracy_noisy": acc_noisy,
                "pred_time_s": pred_time
            })
        except Exception as e:
            robustness_rows.append({
                "model": name,
                "noise_std": std,
                "accuracy_noisy": np.nan,
                "pred_time_s": np.nan
            })

robust_df = pd.DataFrame(robustness_rows)
robust_pivot = robust_df.pivot(index="model", columns="noise_std", values="accuracy_noisy")
cjt.display_dataframe_to_user("Robustness: accuracy at different noise std (digits dataset)", robust_pivot)

# Show noisy sample grids
plt.figure(figsize=(8,3))
for i, std in enumerate(noise_levels):
    plt.subplot(1, len(noise_levels), i+1)
    noisy = add_gaussian_noise(X_hold[:9], std=std)
    grid = np.block([[noisy[r*3+c].reshape(8,8) for c in range(3)] for r in range(3)])
    plt.imshow(grid, cmap='gray', interpolation='nearest')
    plt.title(f"noise std={std}")
    plt.axis('off')
plt.suptitle("Examples of noisy holdout images (3x3 grids)")
plt.show()

# Classification report for best model
best_model_name = results_df.index[0]
best_model = models[best_model_name]
print("Best model:", best_model_name)
if best_model_name.endswith("_flat") or best_model_name in ("knn_raw","bag_dt_flat","rf_flat"):
    Xhold_for_pred = X_hold
else:
    Xhold_for_pred = Xhold_fe_scaled
pred_clean = best_model.predict(Xhold_for_pred)
print("Classification report on clean holdout:")
print(classification_report(y_hold, pred_clean, zero_division=0))

# Noisy report std=0.2
X_hold_noisy_02 = add_gaussian_noise(X_hold, std=0.2)
fe_noisy_02 = compute_features(X_hold_noisy_02)
Xhold_pca20_noisy_02 = pca20.transform(X_hold_noisy_02)
Xhold_fe_noisy_02 = np.hstack([Xhold_pca20_noisy_02, fe_noisy_02])
Xhold_fe_noisy_02_scaled = scaler.transform(Xhold_fe_noisy_02)

if best_model_name.endswith("_flat") or best_model_name in ("knn_raw","bag_dt_flat","rf_flat"):
    Xpred_noisy = X_hold_noisy_02
else:
    Xpred_noisy = Xhold_fe_noisy_02_scaled
pred_noisy_02 = best_model.predict(Xpred_noisy)
print("Classification report on noisy holdout (std=0.2):")
print(classification_report(y_hold, pred_noisy_02, zero_division=0))

