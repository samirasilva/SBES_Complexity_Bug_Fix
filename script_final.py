import os
import pandas as pd
from git import Repo
from radon.complexity import cc_visit
from radon.visitors import Function
from functools import lru_cache
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

import statsmodels.api as sm

# -------------------------
# REPOSITÓRIOS
# -------------------------
REPOS = [
    "repos/requests",
    "repos/flask",
    "repos/click",
    "repos/scrapy",
    "repos/celery",
    "repos/pylint"
]

SINCE_DATE = "2022-01-01"

# -------------------------
# BUG KEYWORDS
# -------------------------
BUG_KEYWORDS = [
    "fix bug",
    "bug fix",
    "fixes",
    "bug",
    "error",
    "defect",
    "fault"
]

# -------------------------
# AUTHORS
# -------------------------
@lru_cache(maxsize=None)
def get_file_authors(repo_path, file_path):
    repo = Repo(repo_path)

    try:
        output = repo.git.log(
            "--follow",
            f"--since={SINCE_DATE}",
            "--format=%aE",
            "--",
            file_path
        )
        return len(set(output.splitlines()))
    except Exception:
        return 0

# -------------------------
# PEGAR ARQUIVOS PYTHON
# -------------------------
def get_all_files(repo_path):
    repo = Repo(repo_path)
    files = repo.git.ls_files().split("\n")

    excluded_dirs = (
        "tests/",
        "test/",
        "doc/",
        "docs/",
        "examples/",
        "example/",
    )

    return [
        f for f in files
        if f.endswith(".py")
        and not f.startswith(excluded_dirs)
    ]

# -------------------------
# COMPLEXIDADE CÍCLOMÁTICA
# -------------------------
def get_complexity(repo_path, files):
    results = {}

    for f in files:
        full_path = os.path.join(repo_path, f)

        try:
            with open(full_path, "r", encoding="utf-8") as file:
                code = file.read()

            loc = len(code.splitlines())
            blocks = cc_visit(code)

            complexities = [
                b.complexity
                for b in blocks
                if isinstance(b, Function)
            ]

            if complexities:
                total = sum(complexities)
                avg = round(total / len(complexities), 3)
                max_c = max(complexities)
            else:
                total = 0
                avg = 0
                max_c = 0

            results[f] = {
                "total": total,
                "avg": avg,
                "max": max_c,
                "loc": loc
            }

        except Exception as e:
            print(f"Erro ao processar {f}: {e}")
            results[f] = {
                "total": 0,
                "avg": 0,
                "max": 0,
                "loc": 0
            }

    return results

# -------------------------
# MÉTRICAS DO GIT
# -------------------------
def get_git_metrics(repo_path, files):
    repo = Repo(repo_path)

    bug = {}
    churn = {}
    commits = {}

    file_set = set(files)

    for commit in repo.iter_commits(since=SINCE_DATE):
        msg = commit.message.lower()

        is_bugfix = (
            any(k in msg for k in BUG_KEYWORDS)
            and 3 < len(msg.split()) < 50
        )

        for f, change in commit.stats.files.items():

            if f not in file_set:
                continue

            if is_bugfix:
                bug[f] = bug.get(f, 0) + 1

            churn[f] = (
                churn.get(f, 0)
                + change.get("insertions", 0)
                + change.get("deletions", 0)
            )

            commits.setdefault(f, set())
            commits[f].add(commit.hexsha)

    commits = {k: len(v) for k, v in commits.items()}

    authors = {}
    for f in file_set:
        authors[f] = get_file_authors(repo_path, f)

    return bug, churn, commits, authors

# -------------------------
# DATASET FINAL
# -------------------------
def build_dataset(repo_path):
    files = get_all_files(repo_path)

    complexity = get_complexity(repo_path, files)
    bugs, churn, commits, authors = get_git_metrics(repo_path, files)

    data = []

    for f in files:
        comp = complexity.get(f, {})

        data.append({
            "repo": repo_path,
            "file": f,
            "complexity_total": comp.get("total", 0),
            "complexity_avg": comp.get("avg", 0),
            "complexity_max": comp.get("max", 0),
            "loc": comp.get("loc", 0),
            "bug_fixes": bugs.get(f, 0),
            "churn": churn.get(f, 0),
            "commits": commits.get(f, 0),
            "authors_commit": authors.get(f, 0)
        })

    return data

# -------------------------
# RUN EXPERIMENT
# -------------------------
all_data = []

for repo in REPOS:
    print("Processando:", repo)
    all_data.extend(build_dataset(repo))

df = pd.DataFrame(all_data)

# -------------------------
# VALIDAÇÕES
# -------------------------
print("\nArquivos por repositório:")
print(df.groupby("repo").size())

print("\nResumo das métricas de complexidade:")
print(df[["complexity_total", "complexity_avg", "complexity_max"]].describe())

print("\nMaior complexity_max:")
print(df["complexity_max"].max())

print("\nTop 10 arquivos por complexity_max:")
print(df.sort_values("complexity_max", ascending=False).head(10))

# -------------------------
# SALVAR
# -------------------------
df.to_excel("results_full.xlsx", index=False)

print("\nDataset salvo em: results_full.xlsx")
print(df.head())

# -------------------------
# LIMPEZA
# -------------------------
df_clean = df.copy()

df_clean = df_clean[df_clean["loc"] > 0]
df_clean = df_clean[df_clean["complexity_total"] > 0]

# -------------------------
# LOG TRANSFORM
# -------------------------
df_clean["log_bug_fixes"] = np.log1p(df_clean["bug_fixes"])
df_clean["log_churn"] = np.log1p(df_clean["churn"])
df_clean["log_loc"] = np.log1p(df_clean["loc"])
df_clean["log_commits"] = np.log1p(df_clean["commits"])

# -------------------------
# MODELO 1 (complexity_avg)
# -------------------------
X = df_clean[[
    "complexity_avg",
    "log_loc",
    "log_churn",
    "log_commits",
    "authors_commit"
]]

X = sm.add_constant(X)
y = df_clean["log_bug_fixes"]

model = sm.OLS(y, X).fit()

print("\n===== REGRESSION (complexity_avg) =====")
print(model.summary())

# Calcula VIF
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [
    variance_inflation_factor(X.values, i)
    for i in range(X.shape[1])
]
print(vif_data)


# -------------------------
# MODELO 2 (complexity_max)
# -------------------------
X2 = df_clean[[
    "complexity_max",
    "log_loc",
    "log_churn",
    "log_commits",
    "authors_commit"
]]

X2 = sm.add_constant(X2)

model2 = sm.OLS(y, X2).fit()

print("\n===== REGRESSION (complexity_max) =====")
print(model2.summary())

# Calcula VIF
vif_data = pd.DataFrame()
vif_data["Variable"] = X2.columns
vif_data["VIF"] = [
    variance_inflation_factor(X2.values, i)
    for i in range(X2.shape[1])
]
print(vif_data)


# -------------------------
# CORRELAÇÃO
# -------------------------
from scipy.stats import spearmanr

rho1, p1 = spearmanr(df_clean["complexity_avg"], df_clean["bug_fixes"])
rho2, p2 = spearmanr(df_clean["churn"], df_clean["bug_fixes"])
rho3, p3 = spearmanr(df_clean["complexity_max"], df_clean["bug_fixes"])

print("\n===== SPEARMAN =====")
print(f"complexity_avg vs bug_fixes: rho={rho1:.4f}, p={p1:.6f}")
print(f"complexity_max vs bug_fixes: rho={rho3:.4f}, p={p3:.6f}")
print(f"churn vs bug_fixes:         rho={rho2:.4f}, p={p2:.6f}")

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(6, 8),sharey=True)

# Average complexity
axes[0].scatter(df["complexity_avg"], df["bug_fixes"], alpha=0.25,s=20)
axes[0].set_xlabel("Average Complexity")
axes[0].set_ylabel("Bug-fixing Activity")
axes[0].set_title("")

# Maximum complexity
axes[1].scatter(df["complexity_max"], df["bug_fixes"], alpha=0.25,s=20)
axes[1].set_xlabel("Maximum Complexity")
axes[1].set_ylabel("Bug-fixing Activity")
axes[1].set_title("")

plt.tight_layout()

plt.savefig("scatter_complexity_combined.png", dpi=300, bbox_inches="tight")
plt.close()
