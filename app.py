import pandas as pd
import streamlit as st

from PIL import Image
from func import *

st.set_page_config(page_title='コラージュ画像検索', layout='wide')

_left, center, _right = st.beta_columns([1, 1, 1])

with center:
    q = st.text_input('検索単語')
    if len(q) > 0:
        min_width = 200
        with st.spinner('画像を検索中...'):
            imgs = main(q, min_width)

n_imgs = len(imgs) if len(q) > 0 else 0
n_cols = 10
n_rows, mod = divmod(n_imgs, n_cols)
if mod > 0: n_rows += 1

i = 0
for row in range(n_rows):
    for col in st.beta_columns(n_cols):
        if i > n_imgs: break
        with col:
            st.image(imgs[i], use_column_width=True)
            i += 1
    else: continue
    break
        
