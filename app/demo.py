import streamlit as st
from PIL import Image
import numpy as np
from io import BytesIO
from main import run_classification, run_detection
from detection import remove_sem_label, create_mask, process_sem_label
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
import zipfile
import os

st.set_page_config(layout="wide", page_title="Sem particle Analyzer")

st.write("## Detect and classsify particles in SEM images")
st.write(
    "Try uploading an image"
)
st.sidebar.write("## Upload your SEM Micrographs :gear:")

if 'og_image' not in st.session_state:
    st.session_state['og_image'] = None
if 'removed_image' not in st.session_state:
    st.session_state['removed_image'] = None
if 'mask' not in st.session_state:
    st.session_state['mask'] = None
if 'particles' not in st.session_state:
    st.session_state['particles'] = None

# Download the fixed image
def save_image(img, img_name):
    # buf = BytesIO()
    # img.save(buf, format="PNG")
    # byte_im = buf.getvalue()
    img.save(img_name)
    return img_name

def create_zip(files, zip_name):
    with zipfile.ZipFile(zip_name, 'w') as zipf:
        for file in files:
            zipf.write(file)

def convert_df(df):
    csv_name = './csv_data.csv'
    df.to_csv(csv_name, index=True)
    return csv_name


def on_label_change():
    img = remove_sem_label(np.array(st.session_state.og_image),
                                      crop_h=st.session_state.label_slider)
    _, label = process_sem_label(np.array(st.session_state.og_image),
                                      crop_h=st.session_state.label_slider)
    img = Image.fromarray(img)
    st.session_state.removed_image = img
    imageLocation.image(img)
    labelLocation.image(Image.fromarray(label))
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

th_value = st.sidebar.slider("Remove SEM label:", min_value=0, max_value=150,
                     value=100, key='label_slider')
# on_change=on_label_change)
th_value = st.sidebar.slider("Fix Thrshold value:", min_value=0, max_value=255,
                     value=127, key='th_slider')

tab1, tab2, tab3, tab4 = st.tabs(["Preprocess", "Morphology", "Chemical", "Docs"])

with tab1:

    col1, col2 = st.columns(2)
    if upload is not None:
        print(upload)
        image = Image.open(upload)

        # st.image(image)

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

        pixel_dx, sem_label_img = process_sem_label(np.array(image))
        labelLocation = st.empty()
        st.session_state.label = sem_label_img

        st.sidebar.markdown("\n")

        on_label_change()
        on_threshold_change()


with tab2:
    scale = st.sidebar.number_input("Relative lenght of sem image line",
                                    value=0, step=1)
    dct_button = st.sidebar.button('Continue', key='detect_botton')
    col1, col2 = st.columns([1, 1])
    if upload is not None and scale !=  0:
        image = Image.open(upload)
        particles, props, displays = run_detection(np.array(image),
                                            crop_h= st.session_state.label_slider,
                                            threshold= st.session_state.th_slider,
                                            scale= scale)

        st.session_state.particles = particles
        col1.image(Image.fromarray(displays['display']))

        fig = px.histogram(props, x="area", title="Area distribution of particles")
        col2.plotly_chart(fig, use_container_width=True)
        props_df = pd.DataFrame(props)
        st.dataframe(props_df)


with tab3:
    col1, col2 = st.columns([1, 1])
    col3, col4 = st.columns([1, 1])
    if dct_button:
        display_img = Image.fromarray(displays['display'])
        col1.image(display_img)

        classifications = run_classification(particles)
        df = pd.DataFrame.from_dict(classifications)#.drop('id')
        col2.dataframe(df)

        ## graphics

        class_counts = df['group'].value_counts().reset_index()
        fig = px.bar(class_counts, x='group', y='count',
                     title='Count for particles per group')
        col3.plotly_chart(fig)

        fig = px.pie(df, names='group', title="Groups distribution")
        col4.plotly_chart(fig)

        # prepare data for download.
        props_df['group'] = df["group"]
        props_df["prob"] = df["prob"]

        csv = convert_df(props_df)

        zip_file_name = "downloaded_files.zip"
        create_zip([csv, save_image(display_img, 'detections.png'),
                    save_image(st.session_state.mask, "mask.png")],
                    zip_file_name)

        # Provide a download link for the zip file
        st.sidebar.download_button(label="Download Zip File",
                           data=open(zip_file_name, "rb").read(),
                           file_name= "data.zip",
                            key="download_zip",
                           mime="application/zip"
                           )

        # Cleanup: Remove the zip file after download
        # os.remove(zip_file_name)

with tab4:
    components.html("""<iframe src="https://scribehow.com/embed/SEM_Particle_Analyzer__User_Guide__7of25zfCT2mO6V4Reymr3w" 
    width="100%" height="640" allowfullscreen frameborder="0"></iframe>""",
                    height=640)


