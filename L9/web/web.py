import streamlit as st
import pandas as pd
from scipy import spatial
from sentence_transformers import SentenceTransformer

st.title('Online shop')

DATA_URL = 'https://gist.github.com/ko3a4ok/9e8128ca2917a2b9379a716b50ef621c/raw/0df0af04199f0cfcfe6fb7531f467bbbaf0e183e/cymbal_product_desc.txt'

@st.cache_resource
def get_model():
  return SentenceTransformer('TaylorAI/bge-micro')

data_load_state = st.text('Loading data...')

@st.cache_data
def load_data():
  df = pd.read_fwf(DATA_URL).iloc[:1000, :1]
  df.columns = ['product description']
  model = get_model()
  df['embedding'] = df.apply(lambda row: model.encode(row[0]), axis=1)
  return df[:1000]


data = load_data()
data_load_state.text("Data is loaded!")

txt = st.text_input(label="search", value="",
                    placeholder="What are you looking for today?")

if txt:
  st.subheader('Results:')
  emb = get_model().encode(txt)
  results = data.sort_values(
      by='embedding',
      key=lambda x: x.map(lambda col:  spatial.distance.cosine(emb, col)),
  )[:10]['product description']
  st.write(results)
