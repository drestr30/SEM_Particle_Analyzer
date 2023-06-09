import streamlit as st
from PIL import Image
import numpy as np
from io import BytesIO
from main import run_classification, run_detection
from detection import remove_sem_label, create_mask
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide", page_title="Image Background Remover")

st.write("## Detect and classsify particles in SEM images")
st.write(
    "Try uploading an image"
)
st.sidebar.write("## Upload and download :gear:")

if 'og_image' not in st.session_state:
    st.session_state['og_image'] = None
if 'removed_image' not in st.session_state:
    st.session_state['removed_image'] = None
if 'mask' not in st.session_state:
    st.session_state['mask'] = None

# Download the fixed image
def download_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

def on_label_change():
    img = remove_sem_label(np.array(st.session_state.og_image),
                                      crop_h=st.session_state.label_slider)
    img = Image.fromarray(img)
    st.session_state.removed_image = img
    imageLocation.image(img)
    # return img

def on_threshold_change():
    mask = create_mask(np.array(st.session_state.removed_image),
                                      threshold=st.session_state.th_slider)
    # bboxes, mask = detect_from_mask(mask)
    mask = Image.fromarray(mask)
    st.session_state.mask = mask
    maskLocation.image(mask)
    # return mask

upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg", 'tif'])

tab1, tab2, tab3 = st.tabs(["Preprocess", "Results", "Owl"])

with tab1:
    th_value = st.slider("Remove SEM label:", min_value=0, max_value=150,
                         value=100, key='label_slider')
    # on_change=on_label_change)
    th_value = st.slider("Fix Thrshold value:", min_value=0, max_value=255,
                         value=127, key='th_slider')
    col1, col2 = st.columns(2)
    if upload is not None:
        image = Image.open(upload)
        col1.write("Original Image :camera:")
        imageLocation = col1.empty()
        st.session_state.og_image = image

        # _, props, mask, _ = detect_and_crop(np.array(image))
        img = remove_sem_label(np.array(image))
        mask = create_mask(img)
        mask = Image.fromarray(mask)
        st.session_state.mask = mask

        col2.write("Fixed Image :wrench:")
        maskLocation = col2.empty()

        st.sidebar.markdown("\n")

        on_label_change()
        on_threshold_change()

        st.sidebar.download_button("Download Mask", download_image(mask),
                                   "fixed.png", "image/png")

with tab2:
    dct_button = st.sidebar.button('Continue', key='detect_botton')
    if upload is not None:
        image = Image.open(upload)
        particles, props, mask, display = run_detection(np.array(image),
                                            crop_h= st.session_state.label_slider,
                                            threshold= st.session_state.th_slider)

        col1, col2, col3 = st.columns([2, 1, 1])

        col1.image(Image.fromarray(display))

        if dct_button:
            classifications = run_classification(particles)


            df = pd.DataFrame.from_dict(classifications)#.drop('id')
            col2.dataframe(df)

            class_counts = df['group'].value_counts().reset_index()
            fig = px.bar(class_counts, x='index', y='group',
                         title='Count for particles per group')
            fig.update_layout(height=500, width=250)

            col3.plotly_chart(fig)



            fig = px.pie(df, names='group')
            st.plotly_chart(fig)

            st.dataframe(pd.DataFrame(props))







