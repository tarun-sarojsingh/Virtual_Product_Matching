# Visual Product Matcher (Simple)

A tiny Streamlit app that finds **visually similar products** from a small catalog.
It uses a lightweight feature (perceptual hash + color histogram) so it runs fast on free tiers.

## Features
- Upload an image **or** paste an **image URL**
- See your uploaded image
- Get a list of visually similar products
- Filter results by **similarity score** and **category**
- Catalog of **60 products** with images and basic metadata
- Loading states and basic error handling
- Mobile-responsive UI (Streamlit)

## Tech Stack
- Python, Streamlit
- Pillow, ImageHash, NumPy, pandas, scikit-learn
- Dataset: placeholder images from `picsum.photos` seeded URLs

## Local Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```

The app will open in your browser. If ports are restricted, Streamlit prints a local URL to click.

## Deploy (Free) — Streamlit Community Cloud
1. Create a new public GitHub repo and push these files.
2. Go to https://share.streamlit.io
3. Sign in with GitHub and click **New app**.
4. Pick your repo, set the **main file** to `app.py`, and click **Deploy**.
5. After the first build, your app has a public URL you can share.

## How It Works (short)
- We compute a 128-dim feature per image:
  - 64 bits from **perceptual hash (pHash)**.
  - 64 dims from a simple **RGB histogram** (padded to 64).
- We **L2-normalize** features and use **cosine similarity** for matching.
- For production, consider replacing the feature with a CLIP embedding (e.g., `sentence-transformers/clip-ViT-B-32`)
  to improve semantic matching.

## Data
- `data/products.csv` contains 60 rows with: `id, name, category, price, image_url`.
- Images are generated via `https://picsum.photos/seed/<id>/512/512`, so they always load.

## Notes
- This is intentionally **simple** for an assessment. Accuracy is decent for color/shape similarity.
- You can expand the catalog by adding more rows to `data/products.csv`.
- To precompute and cache features across runs, you can persist `@st.cache_data` results (Streamlit Cloud handles basic caching).
- Basic error handling is included for image loading and inputs.

## Folder Structure
```text
visual-product-matcher/
├── app.py
├── requirements.txt
└── data/
    └── products.csv
```
