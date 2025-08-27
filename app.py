import io
import requests
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError
from sklearn.metrics.pairwise import cosine_similarity
import imagehash

st.set_page_config(page_title="Visual Product Matcher", layout="wide")
st.title("🖼️ Visual Product Matcher")

@st.cache_data(show_spinner=False)
def load_catalog():
    df = pd.read_csv("data/products.csv")
    return df

def safe_open_image_from_bytes(bts):
    try:
        img = Image.open(io.BytesIO(bts)).convert("RGB")
        return img
    except UnidentifiedImageError:
        return None

def fetch_image_from_url(url, timeout=10):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        img = safe_open_image_from_bytes(r.content)
        return img
    except Exception:
        return None

def image_to_feature(img: Image.Image):
    """
    Very lightweight feature: perceptual hash bits + color histogram.
    Returns a 64(phash) + 64(hist) = 128-dim vector.
    """
    # Perceptual hash (64 bits)
    ph = imagehash.phash(img, hash_size=8)  # 8x8 -> 64 bits
    ph_bits = np.array([int(b) for b in bin(int(str(ph), 16))[2:].zfill(64)], dtype=np.float32)

    # Simple color histogram (per channel 4 bins -> 12 bins; expand to 64 with padding)
    arr = np.array(img.resize((256, 256)))  # speed
    hist_r, _ = np.histogram(arr[:, :, 0], bins=4, range=(0, 256), density=True)
    hist_g, _ = np.histogram(arr[:, :, 1], bins=4, range=(0, 256), density=True)
    hist_b, _ = np.histogram(arr[:, :, 2], bins=4, range=(0, 256), density=True)
    hist = np.concatenate([hist_r, hist_g, hist_b], axis=0)
    # pad to 64 dims for simplicity
    if len(hist) < 64:
        hist = np.pad(hist, (0, 64 - len(hist)), mode='constant')
    hist = hist.astype(np.float32)

    feat = np.concatenate([ph_bits, hist], axis=0)  # 128 dims
    # L2 normalize to make cosine similarity meaningful
    norm = np.linalg.norm(feat) + 1e-8
    return feat / norm

@st.cache_data(show_spinner=True)
def build_catalog_features(df: pd.DataFrame):
    feats = []
    for _, row in df.iterrows():
        img = fetch_image_from_url(row["image_url"])
        if img is None:
            feats.append(None)
            continue
        feats.append(image_to_feature(img))
    # Replace None with average vector to keep lengths aligned (basic fallback)
    if any(f is None for f in feats):
        valid = [f for f in feats if f is not None]
        if len(valid) == 0:
            avg = np.zeros(128, dtype=np.float32)
        else:
            avg = np.mean(np.stack(valid), axis=0).astype(np.float32)
        feats = [avg if f is None else f for f in feats]
    return np.stack(feats)

# Sidebar: Inputs
st.sidebar.header("Upload or Link an Image")
uploaded = st.sidebar.file_uploader("Upload a product image", type=["png", "jpg", "jpeg", "webp"])
url_input = st.sidebar.text_input("...or paste an image URL")
top_k = st.sidebar.slider("How many results?", 5, 30, 12, step=1)
threshold = st.sidebar.slider("Minimum similarity score", 0.0, 1.0, 0.55, step=0.01)

df = load_catalog()
category_filter = st.sidebar.multiselect("Filter by category (optional)", sorted(df["category"].unique().tolist()))

with st.status("Preparing catalog…", expanded=False):
    feats = build_catalog_features(df)

col_left, col_right = st.columns([1, 2], gap="large")

# Gather the query image
query_img = None
if uploaded is not None:
    try:
        query_img = Image.open(uploaded).convert("RGB")
    except UnidentifiedImageError:
        st.sidebar.error("Cannot read the uploaded file as an image.")
        query_img = None
elif url_input.strip():
    query_img = fetch_image_from_url(url_input.strip())
    if query_img is None:
        st.sidebar.error("Could not fetch image from the provided URL.")

with col_left:
    st.subheader("Your Image")
    if query_img is not None:
        st.image(query_img, use_container_width=True)
    else:
        st.caption("Upload a file or paste an image URL to start.")

    st.divider()
    st.subheader("About")
    st.markdown(
        "This demo uses a **very lightweight** feature (perceptual hash + color histogram) "
        "to find visually similar items. It's fast and simple, suitable for demos."
    )

with col_right:
    st.subheader("Similar Products")
    if query_img is None:
        st.info("No image yet. Add one from the sidebar.")
    else:
        with st.spinner("Finding similar items…"):
            q_feat = image_to_feature(query_img).reshape(1, -1)
            sims = cosine_similarity(q_feat, feats)[0]  # (N,)

            # Build result dataframe
            res = df.copy()
            res["similarity"] = sims

            # Apply filters
            if category_filter:
                res = res[res["category"].isin(category_filter)]
            res = res[res["similarity"] >= threshold]

            # Sort and trim
            res = res.sort_values("similarity", ascending=False).head(top_k)

            if res.empty:
                st.warning("No results match your filters. Try lowering the threshold or removing filters.")
            else:
                # Show cards in a responsive grid
                cols = st.columns(3, gap="large")
                for i, (_, row) in enumerate(res.iterrows()):
                    with cols[i % 3]:
                        st.markdown(f"**{row['name']}**")
                        st.caption(f"Category: {row['category']} • Price: ${row['price']}")
                        st.image(row["image_url"], use_container_width=True)
                        st.progress(float(row["similarity"]))

st.divider()
with st.expander("ℹ️ Help & Notes"):
    st.markdown(
        """
        **How it works (simple version):**
        - Each catalog image is turned into a small numeric vector using a *perceptual hash* and a *color histogram*.
        - Your query image is turned into the same kind of vector.
        - We compute **cosine similarity** between your image and every catalog item.
        - You can filter by minimum similarity and by category using the sidebar.

        **Why this is simple:** No heavy ML models are downloaded. This keeps the app tiny and deployable on free tiers.
        For better accuracy later, you can swap the feature extractor with CLIP or another vision model.
        """
    )
