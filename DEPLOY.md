# Deployment

## What gets committed

| Path | Size | Why |
|---|---|---|
| `data/processed/live_book_deploy.parquet` | ~20 MB | the live book the app reads |
| `data/processed/pd_curves.parquet` | ~43 KB | PD term structure |
| `data/processed/lgd_table.parquet` | ~5 KB | LGD by grade |

Not committed: the 1.6 GB raw CSV, `model_set.parquet`, `live_book_staged.parquet`.
Those rebuild from `notebooks/01` through `04`. Also not committed: `.env`.

## 1. Git

```powershell
git init
git add .
git status          # confirm no .env, no data/raw, no *.csv
git commit -m "IFRS 9 ECL engine with agentic copilot"
```

Check the size before pushing:

```powershell
git count-objects -vH
```

Should be roughly 21 MB. If it is hundreds of MB, a data file slipped through.

## 2. GitHub

Create an empty **public** repo (Streamlit Community Cloud needs public, or a
linked private repo on a paid plan). Do not add a README or .gitignore, the
repo already has both.

```powershell
git remote add origin https://github.com/YOUR-USERNAME/credit-risk-copilot.git
git branch -M main
git push -u origin main
```

## 3. Streamlit Community Cloud

1. share.streamlit.io, sign in with GitHub
2. **Create app** -> **Deploy a public app from GitHub**
3. Repository: your repo. Branch: `main`. Main file path: `app.py`
4. Before clicking Deploy, open **Advanced settings** -> **Secrets** and paste:

```toml
LLM_PROVIDER = "gemini"
GEMINI_API_KEY = "your-key-here"
ECL_AGENT_MODEL = "gemini-3.5-flash-lite"
LLM_THROTTLE_SECONDS = "4.5"
```

TOML format, quoted values. Not the `KEY=value` form used in `.env`.

5. Deploy. First build takes 3-6 minutes.

## Troubleshooting

**App restarts on load, no error.** Out of memory. Confirm `live_book_deploy.parquet`
is the file being read, not the staged one. `07_build_deploy_artifact.py` reports
the estimated peak; it should be near 107 MB.

**`FileNotFoundError` on a parquet.** The data files were gitignored. Check
`git ls-files data/processed`.

**Engine panel works, chat panel errors.** Secrets are wrong or missing. The
split is deliberate: the ECL figures never touch the LLM, so the left side of
the app works regardless.

**Slow install.** `requirements.txt` includes `kaggle` and `scipy`, needed for
model building but not at runtime. Swap in `requirements-deploy.txt` if you
want a faster build.

## Verify after deploy

- Weighted ECL reads $905.4m
- Stage 3 coverage reads 91.80%, which equals LGD
- Move the hazard slider to 1.5, stressed ECL should read about $1,040.5m
- Ask the chat "which grade drives the provision under adverse" and confirm
  the grounding badge goes green
