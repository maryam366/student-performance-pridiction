"""
=============================================================
  Student Performance Prediction — EDA (Real UCI Dataset)
  Source: UCI ML Repository — Student Performance Data Set
  P. Cortez and A. Silva, 2008
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Style ────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f1117',
    'axes.facecolor':   '#1a1d27',
    'axes.edgecolor':   '#2e3250',
    'axes.labelcolor':  '#c9cde0',
    'xtick.color':      '#8890b0',
    'ytick.color':      '#8890b0',
    'text.color':       '#c9cde0',
    'grid.color':       '#2e3250',
    'grid.linestyle':   '--',
    'font.family':      'monospace',
})
ACCENT = '#7c83ff'; GREEN = '#43e97b'; RED = '#ff6b6b'; ORANGE = '#ffa94d'
PALETTE = [ACCENT, GREEN, RED, ORANGE, '#a29bfe', '#fd79a8']

# ── Load Real Data ────────────────────────────────────────
mat = pd.read_csv('data/student-mat.csv', sep=';')
por = pd.read_csv('data/student-por.csv', sep=';')

# Combine both courses, add a 'course' label
mat['course'] = 'Math'
por['course'] = 'Portuguese'
df = pd.concat([mat, por], ignore_index=True)

# Target: Pass if G3 >= 10
df['outcome'] = df['G3'].apply(lambda x: 'Pass' if x >= 10 else 'Fail')

print("=" * 60)
print("  STUDENT PERFORMANCE — EDA (REAL UCI DATASET)")
print("=" * 60)
print(f"\n📂 Math students    : {len(mat)}")
print(f"📂 Portuguese stu.  : {len(por)}")
print(f"📂 Combined total   : {len(df)}")
print(f"\n🎯 Outcome distribution:")
print(df['outcome'].value_counts())
print(f"\n📋 Missing values: {df.isnull().sum().sum()} (none — clean dataset!)")
print(f"\n📊 Grade Statistics (G3):")
print(df.groupby('course')['G3'].describe().round(2))

# ── Figure 1: Overview ────────────────────────────────────
fig = plt.figure(figsize=(20, 15), facecolor='#0f1117')
fig.suptitle('UCI Student Performance Dataset — EDA Overview', fontsize=20,
             color='white', fontweight='bold', y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# 1. Pass/Fail pie
ax1 = fig.add_subplot(gs[0, 0])
counts = df['outcome'].value_counts()
wedges, texts, autotexts = ax1.pie(
    counts, labels=counts.index, autopct='%1.1f%%',
    colors=[GREEN, RED], startangle=140,
    textprops={'color': 'white', 'fontsize': 12},
    wedgeprops={'edgecolor': '#0f1117', 'linewidth': 2.5})
for at in autotexts: at.set_fontsize(13)
ax1.set_title('Pass / Fail Distribution\n(G3 ≥ 10 = Pass)', color='white', fontsize=12, pad=10)

# 2. G3 distribution by course
ax2 = fig.add_subplot(gs[0, 1:])
for course, color in zip(['Math', 'Portuguese'], [ACCENT, ORANGE]):
    sub = df[df['course'] == course]['G3']
    ax2.hist(sub, bins=21, alpha=0.65, color=color, label=course,
             edgecolor='white', linewidth=0.4, range=(0, 20))
ax2.axvline(10, color=RED, linestyle='--', linewidth=2, label='Pass threshold (10)')
ax2.set_title('Final Grade (G3) Distribution by Course', color='white', fontsize=13)
ax2.set_xlabel('G3 Score (0–20)'); ax2.set_ylabel('Number of Students')
ax2.legend(facecolor='#1a1d27', edgecolor='#2e3250', labelcolor='white')
ax2.grid(True, alpha=0.3)

# 3. Study time vs pass rate
ax3 = fig.add_subplot(gs[1, 0])
study_labels = {1: '<2h', 2: '2-5h', 3: '5-10h', 4: '>10h'}
df['study_label'] = df['studytime'].map(study_labels)
pass_rate = df.groupby('study_label')['outcome'].apply(
    lambda x: (x == 'Pass').mean() * 100).reindex(['<2h', '2-5h', '5-10h', '>10h'])
bars = ax3.bar(pass_rate.index, pass_rate.values,
               color=[ACCENT, GREEN, ORANGE, RED][::-1], edgecolor='#0f1117', width=0.6)
for bar, val in zip(bars, pass_rate.values):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
             f'{val:.1f}%', ha='center', color='white', fontsize=10)
ax3.set_title('Pass Rate by Study Time', color='white', fontsize=12)
ax3.set_ylabel('Pass Rate (%)'); ax3.set_ylim(0, 105)
ax3.grid(True, alpha=0.3, axis='y')

# 4. Failures vs pass rate
ax4 = fig.add_subplot(gs[1, 1])
fail_pass = df.groupby('failures')['outcome'].apply(
    lambda x: (x == 'Pass').mean() * 100)
bars = ax4.bar(fail_pass.index.astype(str), fail_pass.values,
               color=[GREEN, ORANGE, RED, '#8b0000'], edgecolor='#0f1117', width=0.6)
for bar, val in zip(bars, fail_pass.values):
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
             f'{val:.1f}%', ha='center', color='white', fontsize=10)
ax4.set_title('Pass Rate by # Past Failures', color='white', fontsize=12)
ax4.set_xlabel('Past Failures'); ax4.set_ylabel('Pass Rate (%)')
ax4.set_ylim(0, 105)
ax4.grid(True, alpha=0.3, axis='y')

# 5. Higher education aspiration
ax5 = fig.add_subplot(gs[1, 2])
higher_pass = df.groupby('higher')['outcome'].apply(
    lambda x: (x == 'Pass').mean() * 100)
bars = ax5.bar(['No', 'Yes'], higher_pass.values,
               color=[RED, GREEN], edgecolor='#0f1117', width=0.5)
for bar, val in zip(bars, higher_pass.values):
    ax5.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
             f'{val:.1f}%', ha='center', color='white', fontsize=12)
ax5.set_title('Pass Rate: Wants Higher Education?', color='white', fontsize=12)
ax5.set_xlabel('Wants Higher Education')
ax5.set_ylabel('Pass Rate (%)'); ax5.set_ylim(0, 105)
ax5.grid(True, alpha=0.3, axis='y')

# 6. Correlation heatmap
ax6 = fig.add_subplot(gs[2, :])
num_cols = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures',
            'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3']
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, ax=ax6, cmap='coolwarm', center=0,
            annot=True, fmt='.2f', annot_kws={'size': 7},
            linewidths=0.5, linecolor='#0f1117',
            cbar_kws={'shrink': 0.6})
ax6.set_title('Correlation Matrix — All Numerical Features', color='white', fontsize=13)
ax6.tick_params(labelsize=8)

for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
    ax.set_facecolor('#1a1d27')

plt.savefig('outputs/eda_overview.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("\n✅ Saved: outputs/eda_overview.png")

# ── Figure 2: Feature Deep Dive ───────────────────────────
fig2, axes = plt.subplots(2, 3, figsize=(20, 11), facecolor='#0f1117')
fig2.suptitle('Feature Deep Dive — Key Predictors of Student Success', fontsize=18,
              color='white', fontweight='bold')

# G1 → G2 → G3 trend
ax = axes[0, 0]
for outcome, color in zip(['Pass', 'Fail'], [GREEN, RED]):
    sub = df[df['outcome'] == outcome]
    means = [sub['G1'].mean(), sub['G2'].mean(), sub['G3'].mean()]
    ax.plot(['G1 (Term 1)', 'G2 (Term 2)', 'G3 (Final)'], means, 'o-',
            color=color, linewidth=2.5, markersize=9, label=outcome)
    for i, (x, y) in enumerate(['G1', 'G2', 'G3']):
        pass
ax.set_title('Grade Progression by Outcome', color='white', fontsize=12)
ax.set_ylabel('Average Grade (0–20)')
ax.legend(facecolor='#1a1d27', edgecolor='#2e3250', labelcolor='white')
ax.grid(True, alpha=0.3)

# Alcohol vs G3
ax = axes[0, 1]
walc_g3 = df.groupby('Walc')['G3'].mean()
dalc_g3 = df.groupby('Dalc')['G3'].mean()
ax.plot(walc_g3.index, walc_g3.values, 'o-', color=RED, linewidth=2.5,
        markersize=8, label='Weekend Alcohol')
ax.plot(dalc_g3.index, dalc_g3.values, 's-', color=ORANGE, linewidth=2.5,
        markersize=8, label='Weekday Alcohol')
ax.set_title('Alcohol Consumption vs Avg G3', color='white', fontsize=12)
ax.set_xlabel('Consumption Level (1=Low → 5=High)')
ax.set_ylabel('Avg Final Grade')
ax.legend(facecolor='#1a1d27', edgecolor='#2e3250', labelcolor='white')
ax.grid(True, alpha=0.3)

# Absences vs G3
ax = axes[0, 2]
df_clipped = df[df['absences'] <= 40]
for outcome, color in zip(['Pass', 'Fail'], [GREEN, RED]):
    sub = df_clipped[df_clipped['outcome'] == outcome]
    ax.scatter(sub['absences'], sub['G3'], alpha=0.2, c=color, s=20, label=outcome)
ax.axhline(10, color=ORANGE, linestyle='--', linewidth=1.5)
ax.set_title('Absences vs Final Grade', color='white', fontsize=12)
ax.set_xlabel('Absences'); ax.set_ylabel('G3')
ax.legend(facecolor='#1a1d27', edgecolor='#2e3250', labelcolor='white')
ax.grid(True, alpha=0.3)

# Mother's education vs pass rate
ax = axes[1, 0]
edu_labels = {0: 'None', 1: 'Primary', 2: 'Middle', 3: 'Secondary', 4: 'Higher'}
df['medu_label'] = df['Medu'].map(edu_labels)
medu_pass = df.groupby('medu_label')['outcome'].apply(
    lambda x: (x == 'Pass').mean() * 100).reindex(
    ['None', 'Primary', 'Middle', 'Secondary', 'Higher'])
bars = ax.bar(medu_pass.index, medu_pass.values,
              color=PALETTE[:5], edgecolor='#0f1117', width=0.6)
for bar, val in zip(bars, medu_pass.values):
    if not np.isnan(val):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                f'{val:.0f}%', ha='center', color='white', fontsize=9)
ax.set_title("Pass Rate by Mother's Education", color='white', fontsize=12)
ax.set_xlabel("Mother's Education Level")
ax.set_ylabel('Pass Rate (%)'); ax.set_ylim(0, 105)
ax.tick_params(axis='x', rotation=15)
ax.grid(True, alpha=0.3, axis='y')

# Internet access
ax = axes[1, 1]
internet_pass = df.groupby('internet')['outcome'].apply(
    lambda x: (x == 'Pass').mean() * 100)
bars = ax.bar(['No Internet', 'Has Internet'], internet_pass.values,
              color=[RED, GREEN], edgecolor='#0f1117', width=0.5)
for bar, val in zip(bars, internet_pass.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
            f'{val:.1f}%', ha='center', color='white', fontsize=12)
ax.set_title('Pass Rate by Internet Access', color='white', fontsize=12)
ax.set_ylabel('Pass Rate (%)'); ax.set_ylim(0, 105)
ax.grid(True, alpha=0.3, axis='y')

# Math vs Portuguese pass rate
ax = axes[1, 2]
course_pass = df.groupby('course')['outcome'].apply(
    lambda x: (x == 'Pass').mean() * 100)
bars = ax.bar(course_pass.index, course_pass.values,
              color=[ACCENT, ORANGE], edgecolor='#0f1117', width=0.5)
for bar, val in zip(bars, course_pass.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
            f'{val:.1f}%', ha='center', color='white', fontsize=12)
ax.set_title('Pass Rate: Math vs Portuguese', color='white', fontsize=12)
ax.set_ylabel('Pass Rate (%)'); ax.set_ylim(0, 105)
ax.grid(True, alpha=0.3, axis='y')

for ax in axes.flat:
    ax.set_facecolor('#1a1d27')
    ax.tick_params(colors='#8890b0')

plt.tight_layout()
plt.savefig('outputs/feature_insights.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("✅ Saved: outputs/feature_insights.png")

# Print key insights
print("\n📌 KEY EDA FINDINGS (Real Data):")
print(f"   Math pass rate     : {(mat['G3'] >= 10).mean()*100:.1f}%")
print(f"   Portuguese pass rate: {(por['G3'] >= 10).mean()*100:.1f}%")
print(f"   G1–G3 correlation  : {df[['G1','G3']].corr().iloc[0,1]:.3f}")
print(f"   Failures=0 pass rate: {df[df['failures']==0]['outcome'].eq('Pass').mean()*100:.1f}%")
print(f"   Failures≥1 pass rate: {df[df['failures']>=1]['outcome'].eq('Pass').mean()*100:.1f}%")
print(f"   Higher edu aspir.  : {df[df['higher']=='yes']['outcome'].eq('Pass').mean()*100:.1f}% pass rate")
