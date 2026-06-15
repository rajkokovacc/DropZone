import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, roc_curve

GRAFICI_DIR = 'C:/Users/Rajko/Downloads/grafici'
RANDOM_STATE = 42
TEST_SEZONE = ['2021/22', '2022/23', '2023/24', '2024/25']

COUNT_COLS = ['pobede', 'remiji', 'porazi', 'bodovi',
              'dati_golovi', 'primljeni_golovi', 'gol_razlika',
              'sutevi', 'sutevi_u_okvir', 'korneri', 'fauli',
              'zuti_kartoni', 'crveni_kartoni']

PREVEDENE_KOLONE = {
    'pobede_pm': 'pobede/meč', 'remiji_pm': 'remiji/meč', 'porazi_pm': 'porazi/meč',
    'bodovi_pm': 'bodovi/meč', 'dati_golovi_pm': 'dati golovi/meč',
    'primljeni_golovi_pm': 'primljeni golovi/meč', 'gol_razlika_pm': 'gol-razlika/meč',
    'sutevi_pm': 'šutevi/meč', 'sutevi_u_okvir_pm': 'šutevi u okvir/meč',
    'korneri_pm': 'korneri/meč', 'fauli_pm': 'faulovi/meč',
    'zuti_kartoni_pm': 'žuti kartoni/meč', 'crveni_kartoni_pm': 'crveni kartoni/meč',
    'vrednost': 'tržišna vrednost', 'tip_broj': 'promovisan (da/ne)',
}

BOJE = {
    'Logistička regresija': '#2563eb',
    'Random Forest': '#16a34a',
    'Gradient Boosting': '#d97706',
    'MLP': '#dc2626',
}

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 12,
    'axes.titlesize': 15, 'axes.titleweight': 'bold',
    'axes.labelsize': 12, 'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 120, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
})

os.makedirs(GRAFICI_DIR, exist_ok=True)

opstali = pd.read_csv("C:/Users/Rajko/Downloads/mitnop/opstali_u_premijer_ligi.csv")
promovisani = pd.read_csv("C:/Users/Rajko/Downloads/mitnop/promovisani_klubovi_skalirano_korigovano.csv")
sve_sezone = pd.read_csv("C:/Users/Rajko/Downloads/mitnop/sve_sezone_ishodi.csv")

FEATURES = [c + '_pm' for c in COUNT_COLS]
if 'vrednost' in opstali.columns and 'vrednost' in promovisani.columns:
    FEATURES.append('vrednost')
FEATURES.append('tip_broj')

def po_mecu(df):
    df = df.copy()
    for c in COUNT_COLS:
        df[c + '_pm'] = df[c] / df['utakmice']
    return df

def napravi_ulaz(s_ulaz):
    o = opstali[opstali['sezona'] == s_ulaz].copy(); o['tip_broj'] = 0
    p = promovisani[promovisani['sezona'] == s_ulaz].copy(); p['tip_broj'] = 1
    if len(o) == 0 and len(p) == 0:
        return None
    o, p = po_mecu(o), po_mecu(p)
    keep = FEATURES + ['tim']
    return pd.concat([o[keep], p[keep]], ignore_index=True)

def ulazna_sezona(s_ish):
    g1, g2 = s_ish.split('/')
    return f"{int(g1)-1}/{str(int(g2)-1).zfill(2)}"

def napravi_skup(ishod_sezone):
    delovi = []
    for s_ish in ishod_sezone:
        ulaz = napravi_ulaz(ulazna_sezona(s_ish))
        if ulaz is None:
            continue
        ishodi = sve_sezone[sve_sezone['sezona'] == s_ish][['tim', 'ispao']]
        m = ulaz.merge(ishodi, on='tim', how='left').dropna(subset=['ispao'])
        m['ispao'] = m['ispao'].astype(int)
        m['sezona_ishod'] = s_ish
        delovi.append(m)
    return pd.concat(delovi, ignore_index=True)

sve_ishod = sorted(sve_sezone['sezona'].unique())
train_ishod = [s for s in sve_ishod if s not in TEST_SEZONE]
train_df = napravi_skup(train_ishod)
test_df = napravi_skup(TEST_SEZONE)

def napravi_modele():
    return {
        'Logistička regresija': Pipeline([('sc', StandardScaler()),
            ('clf', LogisticRegression(class_weight='balanced', max_iter=1000,
                                       random_state=RANDOM_STATE))]),
        'Random Forest': Pipeline([('sc', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=300, max_depth=4,
                                           min_samples_leaf=5, max_features='sqrt',
                                           class_weight='balanced',
                                           random_state=RANDOM_STATE, n_jobs=-1))]),
        'Gradient Boosting': Pipeline([('sc', StandardScaler()),
            ('clf', GradientBoostingClassifier(n_estimators=150, max_depth=2,
                                               learning_rate=0.05, subsample=0.8,
                                               random_state=RANDOM_STATE))]),
        'MLP': Pipeline([('sc', StandardScaler()),
            ('clf', MLPClassifier(hidden_layer_sizes=(16,), alpha=1e-2, max_iter=2000,
                                  early_stopping=True, validation_fraction=0.15,
                                  random_state=RANDOM_STATE))]),
    }

def top3(df, proba):
    d = df.copy(); d['p'] = proba
    tp = uk = tacc = n = 0
    for s in d['sezona_ishod'].unique():
        ss = d[d['sezona_ishod'] == s]
        izbor = set(ss.sort_values('p', ascending=False).head(3).index)
        stvarno = set(ss[ss['ispao'] == 1].index)
        tp += len(izbor & stvarno); uk += len(stvarno)
        pred = ss.index.isin(izbor); true = (ss['ispao'] == 1).values
        tacc += int((pred & true).sum() + (~pred & ~true).sum()); n += len(ss)
    return tp, uk, (tacc / n if n else 0)

cv_auc = {ime: [] for ime in napravi_modele()}
for ime in napravi_modele():
    for s in sorted(train_df['sezona_ishod'].unique()):
        tr = train_df[train_df['sezona_ishod'] != s]
        te = train_df[train_df['sezona_ishod'] == s]
        mdl = napravi_modele()[ime]
        mdl.fit(tr[FEATURES].values, tr['ispao'].values)
        pr = mdl.predict_proba(te[FEATURES].values)[:, 1]
        if len(set(te['ispao'])) > 1:
            cv_auc[ime].append(roc_auc_score(te['ispao'].values, pr))

rez = {}
proba_t = {}
for ime, mdl in napravi_modele().items():
    mdl.fit(train_df[FEATURES].values, train_df['ispao'].values)
    pr = mdl.predict_proba(test_df[FEATURES].values)[:, 1]
    tp, uk, acc = top3(test_df, pr)
    rez[ime] = {'AUC': roc_auc_score(test_df['ispao'].values, pr),
                'Top3': tp / uk, 'Acc': acc, 'TP': tp,
                'CV_AUC': float(np.mean(cv_auc[ime])), 'model': mdl}
    proba_t[ime] = pr

modeli = list(rez.keys())
najbolji = max(rez, key=lambda k: rez[k]['AUC'])



print(f"\nTRENING: {train_df['sezona_ishod'].nunique()} sezona, {len(train_df)} uzoraka, "
      f"ispalih {int(train_df['ispao'].sum())} ({train_df['ispao'].mean()*100:.1f}%)")
print(f"TEST:    {test_df['sezona_ishod'].nunique()} sezona, {len(test_df)} uzoraka, "
      f"ispalih {int(test_df['ispao'].sum())}")


print("UNAKRSNA VALIDACIJA ")

print(f"\n{'Model':<24}{'CV AUC (prosek)':<18}")
print("-" * 42)
for m in modeli:
    print(f"{m:<24}{rez[m]['CV_AUC']:<18.3f}")


print("FINALNA EVALUACIJA NA TEST SKUPU ")

print(f"\n{'Model':<24}{'AUC-ROC':<11}{'Top-3 (rec@3)':<16}{'Accuracy':<11}{'F1*':<8}")
print("-" * 70)
for m in modeli:
    print(f"{m:<24}{rez[m]['AUC']:<11.3f}{rez[m]['Top3']:<16.3f}"
          f"{rez[m]['Acc']:<11.3f}{rez[m]['Top3']:<8.3f}")



print(f"DETALJ PO SEZONAMA — {najbolji}")

_td = test_df.copy(); _td['p'] = proba_t[najbolji]
for s in sorted(_td['sezona_ishod'].unique()):
    ss = _td[_td['sezona_ishod'] == s].sort_values('p', ascending=False).head(3)
    pog = int(ss['ispao'].sum())
    print(f"\nSEZONA {s} — pogođeno {pog}/3")
    for i, (_, r) in enumerate(ss.iterrows(), 1):
        tip = "promovisan" if r['tip_broj'] == 1 else "opstao"
        ishod = "ISPAO " if r['ispao'] == 1 else "OPSTAO"
        znak = "[OK]" if r['ispao'] == 1 else "[ - ]"
        print(f"  {i}. {r['tim']:<18} p={r['p']:.1%}  ({tip:<10}) -> {ishod} {znak}")


print(f" {najbolji}")

_clf = rez[najbolji]['model'].named_steps['clf']
if hasattr(_clf, 'coef_'):
    s = pd.Series(_clf.coef_[0], index=FEATURES).sort_values(key=abs, ascending=False)
    
    for f, v in s.head(10).items():
        print(f"  {PREVEDENE_KOLONE.get(f, f):<22}{v:+.3f}")
elif hasattr(_clf, 'feature_importances_'):
    s = pd.Series(_clf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print()
    for f, v in s.head(10).items():
        print(f"  {PREVEDENE_KOLONE.get(f, f):<22}{v:.3f}")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
svi = pd.concat([train_df, test_df])
br = [int((svi['ispao'] == 0).sum()), int((svi['ispao'] == 1).sum())]
a1.bar(['Opstali', 'Ispali'], br, color=['#94a3b8', '#dc2626'], width=0.6)
for i, v in enumerate(br):
    a1.text(i, v + 3, f"{v}\n({v/sum(br)*100:.1f}%)", ha='center', fontsize=11)
a1.set_title('Neuravnoteženost klasa')
a1.set_ylabel('Broj uzoraka (tim-sezona)')
a1.set_ylim(0, max(br) * 1.18)

a2.bar(['Trening\n(17 sezona)', 'Test\n(4 sezone)'],
       [len(train_df), len(test_df)], color=['#2563eb', '#f59e0b'], width=0.6)
for i, v in enumerate([len(train_df), len(test_df)]):
    a2.text(i, v + 3, str(v), ha='center', fontsize=11)
a2.set_title('Podela trening / test (vremenska)')
a2.set_ylabel('Broj uzoraka')
a2.set_ylim(0, len(train_df) * 1.18)
fig.suptitle('Pregled podataka', fontsize=16, fontweight='bold', y=1.02)
fig.savefig(os.path.join(GRAFICI_DIR, '01_pregled_podataka.png')); plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 5.5))
x = np.arange(len(modeli)); w = 0.38
auc_v = [rez[m]['AUC'] for m in modeli]
top_v = [rez[m]['Top3'] for m in modeli]
b1 = ax.bar(x - w/2, auc_v, w, label='AUC-ROC', color='#1e3a8a')
b2 = ax.bar(x + w/2, top_v, w, label='Top-3 pogodak (recall@3)', color='#f59e0b')
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.015,
            f"{b.get_height():.2f}", ha='center', fontsize=10)
ax.set_xticks(x); ax.set_xticklabels(modeli, fontsize=11)
ax.set_ylim(0, 1.05); ax.set_ylabel('Vrednost metrike')
ax.set_title('Poređenje modela na test skupu')
ax.legend(loc='upper right', frameon=False)
ax.axhline(0.5, ls='--', lw=1, color='gray', alpha=0.7)
ax.text(len(modeli)-1, 0.52, 'nasumično (AUC=0.5)', fontsize=9, color='gray', ha='right')
fig.savefig(os.path.join(GRAFICI_DIR, '02_poredjenje_modela.png')); plt.close(fig)

fig, ax = plt.subplots(figsize=(7.5, 7))
for m in modeli:
    fpr, tpr, _ = roc_curve(test_df['ispao'].values, proba_t[m])
    ax.plot(fpr, tpr, lw=2.4, color=BOJE[m], label=f"{m} (AUC={rez[m]['AUC']:.2f})")
ax.plot([0, 1], [0, 1], '--', color='gray', lw=1.2)
ax.set_xlabel('Stopa lažnih pozitiva (FPR)')
ax.set_ylabel('Stopa tačnih pozitiva (TPR)')
ax.set_title('ROC krive — test skup')
ax.legend(loc='lower right', frameon=False, fontsize=11)
ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
fig.savefig(os.path.join(GRAFICI_DIR, '03_roc_krive.png')); plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 5.5))
cvv = [rez[m]['CV_AUC'] for m in modeli]
tev = [rez[m]['AUC'] for m in modeli]
b1 = ax.bar(x - w/2, cvv, w, label='CV AUC (trening, LOSO)', color='#64748b')
b2 = ax.bar(x + w/2, tev, w, label='Test AUC', color='#2563eb')
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.015,
            f"{b.get_height():.2f}", ha='center', fontsize=10)
ax.set_xticks(x); ax.set_xticklabels(modeli, fontsize=11)
ax.set_ylim(0, 1.05); ax.set_ylabel('AUC-ROC')
ax.set_title('Unakrsna validacija vs test (provera generalizacije)')
ax.legend(loc='upper right', frameon=False)
fig.savefig(os.path.join(GRAFICI_DIR, '04_cv_vs_test.png')); plt.close(fig)

clf = rez[najbolji]['model'].named_steps['clf']
fig, ax = plt.subplots(figsize=(9, 6.5))
if hasattr(clf, 'coef_'):
    s = pd.Series(clf.coef_[0], index=FEATURES).sort_values(key=abs)
    boje = ['#dc2626' if v > 0 else '#2563eb' for v in s.values]
    ax.barh([PREVEDENE_KOLONE.get(i, i) for i in s.index], s.values, color=boje)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('Koeficijent (standardizovan prostor)')
    ax.set_title(f'Šta utiče na ispadanje — {najbolji}')
    leg = [Patch(color='#dc2626', label='povećava rizik'),
           Patch(color='#2563eb', label='smanjuje rizik')]
    ax.legend(handles=leg, loc='lower right', frameon=False)
else:
    s = pd.Series(clf.feature_importances_, index=FEATURES).sort_values()
    ax.barh([PREVEDENE_KOLONE.get(i, i) for i in s.index], s.values, color='#2563eb')
    ax.set_xlabel('Važnost'); ax.set_title(f'Važnost karakteristika — {najbolji}')
fig.savefig(os.path.join(GRAFICI_DIR, '05_vaznost_karakteristika.png')); plt.close(fig)

td = test_df.copy(); td['p'] = proba_t[najbolji]
sez = sorted(td['sezona_ishod'].unique())
fig, axes = plt.subplots(1, len(sez), figsize=(4 * len(sez), 5.2))
for ax, s in zip(axes, sez):
    ss = td[td['sezona_ishod'] == s].sort_values('p', ascending=False).head(3)
    pogodak = int(ss['ispao'].sum())
    y = np.arange(3)[::-1]
    boje = ['#16a34a' if r == 1 else '#dc2626' for r in ss['ispao']]
    ax.barh(y, ss['p'].values, color=boje, height=0.6)
    for yi, (_, r) in zip(y, ss.iterrows()):
        ax.text(0.02, yi, f"{r['tim']}", va='center', fontsize=10,
                color='white', fontweight='bold')
        ax.text(r['p'] + 0.01, yi, f"{r['p']:.0%}", va='center', fontsize=9)
    ax.set_title(f"{s}\npogođeno {pogodak}/3", fontsize=12)
    ax.set_xlim(0, 1.05); ax.set_yticks([]); ax.set_xticks([0, 0.5, 1])
    ax.set_xlabel('verovatnoća')
leg = [Patch(color='#16a34a', label='tačno (stvarno ispao)'),
       Patch(color='#dc2626', label='promašaj (opstao)')]
fig.legend(handles=leg, loc='upper center', ncol=2, frameon=False,
           bbox_to_anchor=(0.5, 1.06))
fig.suptitle(f'Top-3 predviđena za ispadanje po sezonama — {najbolji}',
             fontsize=15, fontweight='bold', y=1.13)
fig.savefig(os.path.join(GRAFICI_DIR, '06_predikcije_po_sezonama.png')); plt.close(fig)

fig, ax = plt.subplots(figsize=(10.5, 2.6)); ax.axis('off')
kolone = ['Model', 'CV AUC', 'Test AUC', 'Top-3 (rec@3)', 'Accuracy']
redovi = []
for m in modeli:
    redovi.append([m, f"{rez[m]['CV_AUC']:.3f}", f"{rez[m]['AUC']:.3f}",
                   f"{rez[m]['Top3']:.3f}", f"{rez[m]['Acc']:.3f}"])
tab = ax.table(cellText=redovi, colLabels=kolone, loc='center', cellLoc='center',
               colWidths=[0.30, 0.16, 0.16, 0.20, 0.16])
tab.auto_set_font_size(False); tab.set_fontsize(11); tab.scale(1, 1.7)
for j in range(len(kolone)):
    c = tab[0, j]; c.set_facecolor('#1e3a8a'); c.set_text_props(color='white', fontweight='bold')
best_row = modeli.index(najbolji) + 1
for j in range(len(kolone)):
    tab[best_row, j].set_facecolor('#dbeafe')
ax.set_title('Rezultati poređenja modela', fontweight='bold', pad=18)
fig.savefig(os.path.join(GRAFICI_DIR, '07_tabela_rezultata.png')); plt.close(fig)