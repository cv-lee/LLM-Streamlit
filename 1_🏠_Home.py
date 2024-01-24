import os
import warnings
import streamlit as st
from PIL import Image
from config.config import load_session_state
from pages import *

warnings.filterwarnings("ignore")


# Config
os.environ["TOKENIZERS_PARALLELISM"] = "false"

config_path = "./config/config.json"
secret_path = "./config/secrets/secret.json"
load_session_state(config_path, secret_path)

st.set_page_config(
    page_title="AI-Playground",
    page_icon="🛝",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Header
st.title("AI Playground")
st.subheader("", divider="rainbow")


# Content
image = Image.open(os.path.join(os.path.dirname(__file__), "assets", "home.png"))
st.image(image, width=800)
st.subheader("", divider="rainbow")

# Footer
_, _, _, _, _, _, logo = st.columns(7)
with logo:
    image = Image.open(
        os.path.join(os.path.dirname(__file__), "assets", "fsec_icon.jpg")
    )
    st.image(image, width=150)
