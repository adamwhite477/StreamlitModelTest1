import streamlit as st
from fastai.vision.all import *
from PIL import Image
model = load_learner("model1.pkl")

st.set_page_config(page_title="Genre Predictor", page_icon=":microphone:")

# ---- HEADER -----
with st.container():
    # st.subheader("Hello Test")
    st.title("Test out my model!")
    st.write("This a model that tries to predict the genre of the image it is given")
    st.write("Genres it tries to determine: [Rap, Country, Rock]")
    uploaded_img = st.file_uploader("Upload Image Here", type="jpg", accept_multiple_files=False)

if uploaded_img is not None:
    img = Image.open(uploaded_img)
    st.image(img)
    genre,_,probs = model.predict(img)
    with st.container():
        st.write("This is a {} album".format(genre))
        st.write("It has a confidence of {:.4f}".format(probs[0]))
