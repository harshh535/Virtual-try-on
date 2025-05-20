import streamlit as st

def set_page_config():
    st.set_page_config(page_title="Virtual Try-On", page_icon="👕", layout="wide")
    
    # Custom CSS Styling
    st.markdown("""
        <style>
        /* Main background */
        .stApp {
            background-color: white;
        }
        </style>
    """, unsafe_allow_html=True) 