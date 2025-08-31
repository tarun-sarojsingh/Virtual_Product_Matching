import io
import requests
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError
from sklearn.metrics.pairwise import cosine_similarity
import imagehash

# Set up the page with a friendly title and layout
st.set_page_config(page_title="Picture-Based Product Finder", layout="wide")
st.title("Visual Product Matcher Build ")

# Cache the product catalog to avoid reloading it
@st.cache_data(show_spinner=False)
def load_product_catalog():
    products = pd.read_csv("data/products.csv")
    return products

# Safely open an image from raw bytes
def try_opening_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return image
    except UnidentifiedImageError:
        return None

# Grab an image from a URL
def get_image_from_url(url, timeout=10):
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        image = try_opening_image(response.content)
        return image
    except Exception:
        return None

# Turn an image into a compact feature vector for comparison
def image_to_vector(image: Image.Image):
    """
    Creates a simple feature vector using perceptual hash and color histogram.
    Returns a 128-dimensional vector (64 for hash + 64 for histogram).
    """
    # Generate a perceptual hash (64 bits)
    perceptual_hash = imagehash.phash(image, hash_size=8)  # 8x8 grid gives 64 bits
    hash_bits = np.array([int(bit) for bit in bin(int(str(perceptual_hash), 16))[2:].zfill(64)], dtype=np.float32)

    # Create a simple color histogram (4 bins per channel, padded to 64 dimensions)
    image_array = np.array(image.resize((256, 256)))  # Resize for speed
    red_hist, _ = np.histogram(image_array[:, :, 0], bins=4, range=(0, 256), density=True)
    green_hist, _ = np.histogram(image_array[:, :, 1], bins=4, range=(0, 256), density=True)
    blue_hist, _ = np.histogram(image_array[:, :, 2], bins=4, range=(0, 256), density=True)
    color_hist = np.concatenate([red_hist, green_hist, blue_hist], axis=0)
    # Pad to 64 dimensions for consistency
    if len(color_hist) < 64:
        color_hist = np.pad(color_hist, (0, 64 - len(color_hist)), mode='constant')
    color_hist = color_hist.astype(np.float32)

    # Combine hash and histogram into a 128-dimensional vector
    feature_vector = np.concatenate([hash_bits, color_hist], axis=0)
    # Normalize to make similarity comparisons meaningful
    norm = np.linalg.norm(feature_vector) + 1e-8
    return feature_vector / norm

# Cache the feature vectors for the catalog
@st.cache_data(show_spinner=True)
def prepare_catalog_vectors(products: pd.DataFrame):
    vectors = []
    for _, product in products.iterrows():
        image = get_image_from_url(product["image_url"])
        if image is None:
            vectors.append(None)
            continue
        vectors.append(image_to_vector(image))
    
    # Replace missing vectors with the average to keep things aligned
    if any(v is None for v in vectors):
        valid_vectors = [v for v in vectors if v is not None]
        if len(valid_vectors) == 0:
            average_vector = np.zeros(128, dtype=np.float32)
        else:
            average_vector = np.mean(np.stack(valid_vectors), axis=0).astype(np.float32)
        vectors = [average_vector if v is None else v for v in vectors]
    return np.stack(vectors)

# Sidebar: Let users upload an image or paste a URL
st.sidebar.header("Pick Your Image")
uploaded_image = st.sidebar.file_uploader("Upload a product picture", type=["png", "jpg", "jpeg", "webp"])
url_input = st.sidebar.text_input("...or paste an image link")
num_results = st.sidebar.slider("How many matches to show?", 5, 30, 12, step=1)
min_similarity = st.sidebar.slider("Minimum match quality", 0.0, 1.0, 0.55, step=0.01)

# Load the product catalog
products = load_product_catalog()
category_options = sorted(products["category"].unique().tolist())
selected_categories = st.sidebar.multiselect("Filter by category (optional)", category_options)

# Prepare the catalog's feature vectors
with st.status("Getting catalog ready…", expanded=False):
    catalog_vectors = prepare_catalog_vectors(products)

# Set up two columns for the layout
left_column, right_column = st.columns([1, 2], gap="large")

# Handle the user's query image
user_image = None
if uploaded_image is not None:
    try:
        user_image = Image.open(uploaded_image).convert("RGB")
    except UnidentifiedImageError:
        st.sidebar.error("Couldn't read the uploaded file as an image.")
        user_image = None
elif url_input.strip():
    user_image = get_image_from_url(url_input.strip())
    if user_image is None:
        st.sidebar.error("Couldn't fetch the image from that URL.")

# Left column: Show the user's image and app info
with left_column:
    st.subheader("Your Picture")
    if user_image is not None:
        st.image(user_image, use_column_width=True)
    else:
        st.caption("Upload a picture or paste an image link to get started.")

    st.divider()
    st.subheader("About This App")
    st.markdown(
        "This tool uses a **super lightweight** approach (perceptual hash + color histogram) "
        "to find products that look similar."
    )

# Right column: Show matching products
with right_column:
    st.subheader("Similar Products")
    if user_image is None:
        st.info("No image yet. Add one using the sidebar.")
    else:
        with st.spinner("Searching for matches…"):
            query_vector = image_to_vector(user_image).reshape(1, -1)
            similarities = cosine_similarity(query_vector, catalog_vectors)[0]

            # Build the results table
            results = products.copy()
            results["match_score"] = similarities

            # Apply filters
            if selected_categories:
                results = results[results["category"].isin(selected_categories)]
            results = results[results["match_score"] >= min_similarity]

            # Sort and limit the results
            results = results.sort_values("match_score", ascending=False).head(num_results)

            if results.empty:
                st.warning("No matches found. Try lowering the similarity threshold or removing category filters.")
            else:
                # Display results in a grid
                columns = st.columns(3, gap="large")
                for i, (_, product) in enumerate(results.iterrows()):
                    with columns[i % 3]:
                        st.markdown(f"**{product['name']}**")
                        st.caption(f"Category: {product['category']} • Price: Rs{product['price']}")
                        st.image(product["image_url"], use_column_width=True)
                        st.progress(float(product["match_score"]))

# Add a help section
st.divider()
with st.expander("ℹ️ Help & Notes"):
    st.markdown(
        """
        **How it works (in simple terms):**
        - Each product image is turned into a small numeric vector using a *perceptual hash* and a *color histogram*.
        - Your image is converted into a similar vector.
        - We measure **cosine similarity** to find the closest matches in the catalog.
        - Use the sidebar to filter by similarity score or product category.

        **Why this is lightweight:** No bulky ML models are used, so the app stays small and runs on free platforms.
        """
    )
