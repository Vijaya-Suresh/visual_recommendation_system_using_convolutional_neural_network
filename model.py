# Importing Required Libraries
import streamlit as st
from sklearn.preprocessing import StandardScaler
import streamlit as st
import pandas as pd
import numpy as np
import cv2
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import cv2
import os
import tensorflow as tf
import keras
from keras import Model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from keras.layers import GlobalMaxPooling2D

# Set Page Config
st.set_page_config("Visual Recommendation System", page_icon="👠", layout="centered")
st.title('Fashion Recommender System')

# Loading the dataset
Data = r"D:\Mini_Project\fashion-dataset"

df = pd.read_csv(r"D:\Mini_Project\fashion-dataset\styles.csv", nrows=7000, on_bad_lines='skip')
df_embs = pd.read_csv('embeddings_7k.csv', index_col=0)

df['image'] = df.apply(lambda row: str(row['id']) + ".jpg", axis=1)
df = df.reset_index(drop=True)

# Input Shape
img_width, img_height, _ = 224, 224, 3

# Pre-Trained Model
base_model = ResNet50(weights='imagenet',
                      include_top=False,
                      input_shape = (img_width, img_height, 3))
base_model.trainable = False

# Add Layer Embedding
model = keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

def plot_figures(figures, *args, figsize=(16, 8), **kwargs):
    num_figures = len(figures)
    fig, axes = plt.subplots(1, num_figures, figsize=figsize)

    for ax, (title, img) in zip(axes, figures.items()):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def img_path(img):
    return os.path.join(Data, "images", img)

def load_image(img, resized_fac=0.7):
    img_path = os.path.join(Data, "images", img)
    img = cv2.imread(img_path)
    w, h, _ = img.shape
    resized = cv2.resize(img, (int(h * resized_fac), int(w * resized_fac)), interpolation=cv2.INTER_CUBIC)
    return resized

def display_images(file_indices, img_files_list, top_n=5):
    for file in file_indices[:top_n]:
        img_name = img_files_list.iloc[file]
        img_path = os.path.join(Data, "images", img_name)  # Assuming images are in the 'images' directory

        # Use st.image to show images in Streamlit
        st.image(cv2.cvtColor(load_image(img_path), cv2.COLOR_BGR2RGB), caption=f'Product {file + 1}', use_column_width=False, width=200)

def get_recommender(idx, df, indices, top_n=5):
    try:
        sim_idx = indices.loc[idx]
    except KeyError:
        print(f"Index {idx} not found in the 'indices' object.")
        return pd.Index([]), []

    # The rest of your code remains the same...
    sim_scores = list(enumerate(cosine_sim[sim_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]
    idx_rec = [i[0] for i in sim_scores]
    idx_sim = [i[1] for i in sim_scores]

    return indices.iloc[idx_rec], idx_sim

from sklearn.metrics.pairwise import pairwise_distances

# Calcule DIstance Matriz
cosine_sim = 1-pairwise_distances(df_embs, metric='cosine')

indices = pd.Series(range(len(df)), index=df.index)

gender_indices = {
    'Men': df[df['gender'] == 'Men'].index,
    'Women': df[df['gender'] == 'Women'].index,
    'Boys': df[df['gender'] == 'Boys'].index,
    'Unisex': df[df['gender'] == 'Unisex'].index,
    'Girls': df[df['gender'] == 'Girls'].index,
}

# Gender selection using Streamlit
gender_options = list(gender_indices.keys())
use_gender = st.sidebar.checkbox("Use Gender and Indices", value=True)
if use_gender:
    selected_gender = st.sidebar.selectbox("Select Gender", gender_options)
    st.write(f"Selected Gender: {selected_gender}")
    
    indicess = gender_indices[selected_gender]
    selected_index = st.selectbox('Select Index:', indicess, format_func=lambda x: str(x))
    selected_index = int(selected_index)

    ref_img = load_image(df.iloc[selected_index].image)
    st.image(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB), caption='Reference Image', use_column_width=False,width=250)

    # Number of recommendations
    num_recommendations = st.number_input('Number of Recommendations:', min_value=1, value=5)

    # Button to trigger recommendations
    if st.button('Recommend'):
        # Add loading spinner
        with st.spinner('Recommendation in progress...'):
            idx_rec, idx_sim = get_recommender(selected_index, df, indices, top_n=num_recommendations)
            figures = {f'Product {i + 1}': load_image(df.iloc[i].image) for i in idx_rec}
            st.empty()
            st.subheader('Recommended Products')
            images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in figures.values()]
            st.image(images, caption=list(figures.keys()), use_column_width=False,width=200)
    else:
        st.info('Click the "Recommend" button to get product recommendations.')
else:
    uploaded_image = st.sidebar.file_uploader("Upload an image (PNG, JPG)", type=["png", "jpg"])
    
    if uploaded_image is not None:
    # Save the uploaded image temporarily
        temp_image_path = "temp_image.jpg"
    
        try:
            with open(temp_image_path, "wb") as temp_image:
                temp_image.write(uploaded_image.read())
        except Exception as e:
            st.error(f"Error saving uploaded image: {str(e)}")

    # Your image processing code for the unseen image
        input_img_path = temp_image_path  
        input_img = Image.open(input_img_path).resize((224,224))
        input_img_array = np.array(input_img)

        if input_img_array.shape[-1] == 4:
            input_img_array = input_img_array[:, :, :3]

        expand_input_img = np.expand_dims(input_img_array, axis=0)
        preprocessed_input_img = preprocess_input(expand_input_img)
        result_to_resnet_input = model.predict(preprocessed_input_img)
        flatten_result_input = result_to_resnet_input.flatten()

        st.image(np.asarray(input_img), caption='Reference Image', use_column_width=False,width=350)

# Convert df_embs to a sparse matrix
        sparse_embs = csr_matrix(df_embs.values)

# Calculate Cosine Similarity with the input image features
        cosine_sim_input = cosine_similarity(flatten_result_input.reshape(1, -1), sparse_embs)

        # Number of recommended images input
        num_recommendations = st.slider('Select the number of recommended images', min_value=1, max_value=10, value=5)

# Get indices of top similar images
        top_similar_indices = np.argsort(cosine_sim_input[0])[::-1][:num_recommendations]

# Display recommended images in Streamlit
        st.subheader(f'Recommended Products for Uploaded Image')
        for file in top_similar_indices:
            img_name = df['image'].iloc[file]
            img_path = os.path.join(Data, "images", img_name)
            st.image(cv2.cvtColor(load_image(img_path), cv2.COLOR_BGR2RGB), caption=f'Recommended Product', use_column_width=False,width=250)