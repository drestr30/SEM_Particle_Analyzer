import streamlit as st
from PIL import Image
import numpy as np
from io import BytesIO
from main import run_detection_classification
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide", page_title="Image Background Remover")

st.write("## Detect and classsify particles in SEM images")
st.write(
    ":dog: Try uploading an image:grin:"
)
st.sidebar.write("## Upload and download :gear:")

# Download the fixed image
def download_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im


def fix_image(upload):
    image = Image.open(upload)
    col1.write("Original Image :camera:")
    col1.image(image)

    clasifications, mask = run_detection_classification(np.array(image))
    mask = Image.fromarray(mask)
    col2.write("Fixed Image :wrench:")
    col2.image(mask)
    st.sidebar.markdown("\n")
    st.sidebar.download_button("Download fixed image", download_image(mask), "fixed.png", "image/png")

    df = pd.DataFrame.from_dict(clasifications)
    fig = px.pie(df, names='label')
    st.plotly_chart(fig)

col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg", 'tif'])

if my_upload is not None:
    fix_image(upload=my_upload)

