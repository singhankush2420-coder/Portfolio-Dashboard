# Portfolio_Dashboard

This repository contains a Streamlit app `app.py` (PortfolioIQ: Risk, Performance & Attribution). The repository was prepared for deployment on Streamlit Community Cloud.

---

**What I added**
- `requirements.txt` — lists Python packages the app imports.
- `.streamlit/config.toml` — Streamlit server and theme settings.
- `runtime.txt` — (optional) pins Python version for reproducible environment.

**Detected external dependencies (from `app.py`)**
- `streamlit`
- `pandas`
- `numpy`
- `yfinance`
- `matplotlib`
- `plotly`
- `scipy`
- `reportlab`
- `streamlit-autorefresh`
- `requests`

---

## Run locally

1. Create a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

---

## Deploy to Streamlit Community Cloud (quick steps)

1. Push this repository to GitHub (if not already):

```bash
git init
git add .
git commit -m "Initial commit: add app and deployment files"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

2. Go to https://streamlit.io/cloud (Streamlit Community) and sign in with GitHub.
3. Click "New app" → choose the repository, branch (`main`) and set the file path to `app.py`.
4. Deploy. Streamlit Cloud will install packages from `requirements.txt` and run the app.

Notes:
- If your app needs API keys or secrets, do NOT commit them. Use the Streamlit Cloud Secrets manager (on the app page, choose "Settings" → "Secrets") or keep a local `.streamlit/secrets.toml` (add to `.gitignore`).
- If you use large data files, avoid storing them in the repo; use external storage (S3, Google Drive, etc.) or Git LFS.
- If deployment fails due to a missing package, add it to `requirements.txt` and re-deploy.

---

## Useful debugging tips
- Check the app logs in the Streamlit Cloud app page (Logs) for import or runtime errors.
- If you see CORS or port issues locally, `.streamlit/config.toml` sets `enableCORS=false` and `headless=true` for Cloud-friendly behavior.

If you'd like, I can:
- Add a minimal `.gitignore` and `LICENSE`.
- Pin specific package versions in `requirements.txt` using `pip freeze` from your dev environment.
- Inspect `app.py` further for any local file paths or unsupported modules and fix them.
