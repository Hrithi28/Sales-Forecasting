# SalesForecasting_Hrithika

## What's in this folder
- `analysis.ipynb` — Tasks 1–6, fully executed (EDA, decomposition, SARIMA/Prophet/XGBoost forecasts + comparison, category/region forecasts, anomaly detection, K-Means demand segmentation).
- `train.csv`, `vgsales.csv` — the two datasets used.
- `app.py` — Task 7 Streamlit dashboard (4 pages: Sales Overview, Forecast Explorer, Anomaly Report, Product Demand Segments).
- `requirements.txt` — all libraries needed to run the notebook and the app.
- `summary.docx` — Task 8, the 2-page executive business report.
- `charts/` — every chart from the notebook saved as PNG.

## Run it yourself

```bash
pip install -r requirements.txt
jupyter notebook analysis.ipynb      # to re-run/inspect the analysis
streamlit run app.py                 # to launch the dashboard locally
```

## Two steps I can't do on your behalf (need your accounts)

1. **Google Colab link**: upload `analysis.ipynb` (and `train.csv`, `vgsales.csv` alongside it) to
   Google Drive / Colab, run all cells once there, then use "Share" to get the link required by
   the submission form.
2. **Streamlit Community Cloud deployment**:
   - Push this whole folder to a new **public GitHub repo**.
   - Go to https://share.streamlit.io -> "New app" -> pick that repo -> set the main file to `app.py`.
   - Click Deploy. It will install `requirements.txt` automatically and give you the live link.

Both just need your GitHub/Google login, so they have to happen from your side — everything else
in this folder is complete and tested.

- **Colab Link:** https://colab.research.google.com/drive/1dSI3PoaL0AL6pFfhvqwyKbM5n4YVGomf?usp=sharing
- **Streamlit App:** https://sales-forecasting-bvdg5prnpj628sypzmgxxv.streamlit.app/
