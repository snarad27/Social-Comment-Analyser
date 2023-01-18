import numpy as np
import pandas as pd
import streamlit as st

from scrapping import *
import torch
from tqdm import tqdm
from scrapping import scrap



def main():
    st.title("Social Comment Analyzer")
    html_temp="""
    <div style=""background-color:tomato;padding:10px>
    <h2 style="color:white;text-align:center;"> HateSpeech Detection App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    link=st.text_input("Enter your Link","Type here")
    result=""
    if st.button("Predict"):
        result=scrap(link)
    st.success('The output is {}'.format(result))

    
if __name__=='__main__':
    main()