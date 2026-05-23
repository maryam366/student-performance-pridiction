"""
=================================================================
  Student Performance Prediction — ML Pipeline (Real UCI Data)
  Models: Decision Tree | KNN | Logistic Regression
  Dataset: UCI Student Performance (Math + Portuguese)
=================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score,
                             precision_score, recall_score, f1_score)

plt.rcParams.update({
    'figure.facecolor': '#0f1117', 'axes.facecolor': '#1a1d27',
    'axes.edgecolor': '#2e3250', 'axes.labelcolor': '#c9cde0',
    'xtick.color': '#8890b0', 'ytick.color': '#8890b0',
    'text.color': '#c9cde0', 'grid.color': '#2e3250',
    'grid.linestyle': '--', 'font.family': 'monospace',
})
ACCENT = '#7c83ff'; GREEN = '#43e97b'; RED = '#ff6b6b'; ORANGE = '#ffa94d'

# ══════════════════════════════════════════════════════════════
# 1. LOAD & PREPROCESS REAL DATA
# ══════════════════════════════════════════════════════════════
print("=" * 62)
print("  STUDENT PERFORMANCE PREDICTION — REAL UCI DATASET")
print("=" * 62)

mat = pd.read_csv('data/student-mat.csv', sep=';')
por = pd.read_csv('data/student-por.csv', sep=';')
mat['course'] = 'Math'; por['course'] = 'Portuguese'
df = pd.concat([mat, por], ignore_index=True)

print(f"\n📂 Math students      : {len(mat)}")
print(f"📂 Portuguese students: {len(por)}")
print(f"📂 Total combined     : {len(df)}")

# Target: binary Pass/Fail
df['outcome'] = (df['G3'] >= 10).astype(int)   # 1=Pass, 0=Fail
print(f"\n🎯 Pass: {df['outcome'].sum()} | Fail: {(df['outcome']==0).sum()}")

# Encode categoricals
cat_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus',
            'Mjob', 'Fjob', 'reason', 'guardian',
            'schoolsup', 'famsup', 'paid', 'activities',
            'nursery', 'higher', 'internet', 'romantic', 'course']

df_ml = df.copy()
le = LabelEncoder()
for col in cat_cols:
    df_ml[col] = le.fit_transform(df_ml[col].astype(str))

# Features: use G1 & G2 but NOT G3 (it's used to make the target)
FEATURES = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus',
            'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian',
            'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup',
            'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic',
            'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health',
            'absences', 'G1', 'G2', 'course']

X = df_ml[FEATURES]
y = df_ml['outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\n📊 Train: {X_train.shape} | Test: {X_test.shape}")

# ══════════════════════════════════════════════════════════════
# 2. MODEL PIPELINES
# ══════════════════════════════════════════════════════════════
models = {
    'Decision Tree': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', DecisionTreeClassifier(random_state=42))
    ]),
    'KNN': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', KNeighborsClassifier())
    ]),
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(random_state=42, max_iter=1000))
    ])
}

param_grids = {
    'Decision Tree': {
        'clf__max_depth':         [3, 5, 7, 10, None],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf':  [1, 2, 4],
        'clf__criterion':         ['gini', 'entropy']
    },
    'KNN': {
        'clf__n_neighbors': [3, 5, 7, 9, 11, 15],
        'clf__weights':     ['uniform', 'distance'],
        'clf__metric':      ['euclidean', 'manhattan']
    },
    'Logistic Regression': {
        'clf__C':       [0.01, 0.1, 1, 10, 100],
        'clf__penalty': ['l1', 'l2'],
        'clf__solver':  ['liblinear', 'saga']
    }
}

# ══════════════════════════════════════════════════════════════
# 3. BASELINE
# ══════════════════════════════════════════════════════════════
print("\n🔵 Baseline Accuracy (default hyperparameters):")
baseline = {}
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipe.predict(X_test))
    baseline[name] = acc
    print(f"   {name:<22}: {acc*100:.2f}%")

# ══════════════════════════════════════════════════════════════
# 4. GRIDSEARCHCV TUNING
# ══════════════════════════════════════════════════════════════
print("\n🔍 Running GridSearchCV (5-fold CV)...")
best_models = {}
all_metrics = {}

for name, pipe in models.items():
    print(f"   Tuning {name}...", end=' ', flush=True)
    grid = GridSearchCV(pipe, param_grids[name], cv=5,
                        scoring='f1', n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)
    best_models[name] = grid.best_estimator_

    y_pred = grid.best_estimator_.predict(X_test)
    y_prob = grid.best_estimator_.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall':    recall_score(y_test, y_pred),
        'f1':        f1_score(y_test, y_pred),
        'roc_auc':   roc_auc_score(y_test, y_prob),
        'cv_f1':     cross_val_score(grid.best_estimator_, X, y, cv=5, scoring='f1').mean(),
        'y_pred':    y_pred,
        'y_prob':    y_prob,
        'best_params': grid.best_params_
    }
    all_metrics[name] = metrics
    print(f"✓  Acc={metrics['accuracy']*100:.2f}%  F1={metrics['f1']:.3f}  AUC={metrics['roc_auc']:.3f}")
    print(f"      Best: {grid.best_params_}")

# ══════════════════════════════════════════════════════════════
# 5. DETAILED REPORTS
# ══════════════════════════════════════════════════════════════
print("\n📊 Full Classification Reports:")
for name, m in all_metrics.items():
    print(f"\n  ── {name} ──")
    print(classification_report(y_test, m['y_pred'], target_names=['Fail', 'Pass']))

best_name = max(all_metrics, key=lambda k: all_metrics[k]['f1'])
joblib.dump(best_models[best_name], f"models/best_model_{best_name.replace(' ','_')}.pkl")
print(f"\n💾 Best model saved: {best_name}")

# ══════════════════════════════════════════════════════════════
# 6. VISUALIZATIONS
# ══════════════════════════════════════════════════════════════

# — Model Comparison —
fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor='#0f1117')
fig.suptitle('Model Comparison — Before vs After Hyperparameter Tuning',
             fontsize=15, color='white', fontweight='bold')

names = list(models.keys())
short = ['DT', 'KNN', 'LR']
x = np.arange(len(names)); w = 0.35

ax = axes[0]
b1 = ax.bar(x - w/2, [baseline[n]*100 for n in names], w,
            label='Before Tuning', color=ACCENT+'88', edgecolor='#0f1117')
b2 = ax.bar(x + w/2, [all_metrics[n]['accuracy']*100 for n in names], w,
            label='After Tuning', color=GREEN+'cc', edgecolor='#0f1117')
for bar in [*b1, *b2]:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f'{bar.get_height():.1f}%', ha='center', color='white', fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(short)
ax.set_ylabel('Accuracy (%)'); ax.set_ylim(70, 100)
ax.set_title('Accuracy: Before vs After Tuning', color='white')
ax.legend(facecolor='#1a1d27', edgecolor='#2e3250', labelcolor='white')
ax.grid(True, alpha=0.3, axis='y'); ax.set_facecolor('#1a1d27')

ax = axes[1]
metric_keys = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
xlabels = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
x2 = np.arange(len(metric_keys)); bw = 0.25
for i, (name, color) in enumerate(zip(names, [ACCENT, GREEN, ORANGE])):
    vals = [all_metrics[name][m]*100 for m in metric_keys]
    ax.bar(x2 + (i-1)*bw, vals, bw, label=short[i], color=color+'cc', edgecolor='#0f1117')
ax.set_xticks(x2); ax.set_xticklabels(xlabels, rotation=10)
ax.set_ylabel('Score (%)'); ax.set_ylim(70, 100)
ax.set_title('All Metrics Comparison (Tuned)', color='white')
ax.legend(facecolor='#1a1d27', edgecolor='#2e3250', labelcolor='white')
ax.grid(True, alpha=0.3, axis='y'); ax.set_facecolor('#1a1d27')

plt.tight_layout()
plt.savefig('outputs/model_comparison.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("\n✅ Saved: outputs/model_comparison.png")

# — Confusion Matrices —
fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor='#0f1117')
fig.suptitle('Confusion Matrices — All Models (After Tuning)', fontsize=15, color='white', fontweight='bold')
for ax, (name, m) in zip(axes, all_metrics.items()):
    cm = confusion_matrix(y_test, m['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues',
                annot_kws={'size': 18, 'fontweight': 'bold'},
                xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'],
                linewidths=2, linecolor='#0f1117')
    ax.set_title(f'{name}\nAcc={m["accuracy"]*100:.1f}%  F1={m["f1"]:.3f}', color='white', fontsize=12)
    ax.set_xlabel('Predicted', color='#c9cde0'); ax.set_ylabel('Actual', color='#c9cde0')
    ax.set_facecolor('#1a1d27')
plt.tight_layout()
plt.savefig('outputs/confusion_matrices.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("✅ Saved: outputs/confusion_matrices.png")

# — ROC Curves —
fig, ax = plt.subplots(figsize=(9, 7), facecolor='#0f1117')
ax.set_facecolor('#1a1d27')
for (name, m), color in zip(all_metrics.items(), [ACCENT, GREEN, ORANGE]):
    fpr, tpr, _ = roc_curve(y_test, m['y_prob'])
    ax.plot(fpr, tpr, color=color, linewidth=2.5,
            label=f'{name}  (AUC = {m["roc_auc"]:.3f})')
ax.plot([0,1],[0,1], '--', color='#555577', linewidth=1.5, label='Random Baseline')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves — All Models', color='white', fontsize=15, fontweight='bold')
ax.legend(facecolor='#1a1d27', edgecolor='#2e3250', labelcolor='white', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/roc_curves.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("✅ Saved: outputs/roc_curves.png")

# — Feature Importance —
fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor='#0f1117')
fig.suptitle('Feature Importance Analysis', fontsize=16, color='white', fontweight='bold')

# LR coefficients
ax = axes[0]
lr_coef = best_models['Logistic Regression'].named_steps['clf'].coef_[0]
feat_df = pd.DataFrame({'feature': FEATURES, 'coef': lr_coef})
feat_df = feat_df.reindex(feat_df['coef'].abs().sort_values(ascending=False).index).head(15)
colors_f = [GREEN if c > 0 else RED for c in feat_df['coef']]
ax.barh(feat_df['feature'], feat_df['coef'], color=colors_f, edgecolor='#0f1117')
ax.axvline(0, color='white', linewidth=1.2, linestyle='--')
ax.set_title('Logistic Regression Coefficients\n(Top 15)', color='white', fontsize=12)
ax.set_xlabel('Coefficient Value'); ax.set_facecolor('#1a1d27')
ax.grid(True, alpha=0.3, axis='x')

# DT importances
ax = axes[1]
dt_imp = best_models['Decision Tree'].named_steps['clf'].feature_importances_
imp_df = pd.DataFrame({'feature': FEATURES, 'importance': dt_imp})
imp_df = imp_df.sort_values('importance', ascending=False).head(15)
ax.barh(imp_df['feature'], imp_df['importance'], color=ACCENT+'cc', edgecolor='#0f1117')
ax.set_title('Decision Tree Feature Importances\n(Top 15)', color='white', fontsize=12)
ax.set_xlabel('Importance Score'); ax.set_facecolor('#1a1d27')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('outputs/feature_importance.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("✅ Saved: outputs/feature_importance.png")

# — Final Summary —
print("\n" + "=" * 65)
print("  FINAL RESULTS SUMMARY (Real UCI Dataset)")
print("=" * 65)
print(f"{'Model':<25} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7} {'CV-F1':>7}")
print("-" * 65)
for name, m in all_metrics.items():
    print(f"{name:<25} {m['accuracy']*100:>6.1f}% {m['precision']*100:>6.1f}% "
          f"{m['recall']*100:>6.1f}% {m['f1']:>7.3f} {m['roc_auc']:>7.3f} {m['cv_f1']:>7.3f}")
print("-" * 65)
print(f"\n🏆 Best Model: {best_name}  (F1 = {all_metrics[best_name]['f1']:.3f})")
print("=" * 65)
