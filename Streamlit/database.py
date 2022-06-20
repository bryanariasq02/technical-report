# from multiprocessing.sharedctypes import Value
# import pandas as pd
# import numpy as np
# import pymongo
# import sys
# sys.path.append("C:/Users/luisalexander/Desktop/3BIO-Scientometrics/SCRIPTS/Librerias")
# from importe_datosSiB import importar_datos_SiB, buscar_datos_SiB, convertir_df
# import getpass
import streamlit as st
from PIL import Image
from datos import *
image = Image.open('metrics.png')
# import ast 

# Referencias
# https://docs.microsoft.com/en-us/azure/cosmos-db/sql/create-cosmosdb-resources-portal
# Para ejecutar 
# python -m streamlit run prueba_base_datos.py

st.set_page_config(
    page_title="Database metrics",
    page_icon="📊",
    layout="wide",
)

st.title ("Métricas base de datos cosmoDB mongoDB")
st.markdown('En el presente documento se relacionan algunas metricas tecnicas sobre la subida de datos a cosmosDB y algunas consultas, con el fin de analizar el comportamiento de la herramienta de azure COSMOSDB API para MONGODB ref: https://docs.microsoft.com/en-us/azure/cosmos-db/sql/create-cosmosdb-resources-portal')

# Descarga de los datos del SiB 
st.header('Descarga, limpieza y actualización de los datos del GrupLAC y SiB')

st.metric(label="Base de datos", value="1")

row1_1, row1_2 = st.columns((2, 3))

with row1_1:
    st.subheader('Opción 1:')
    st.metric(label="Colecciones", value="2")
    st.markdown("GrupLAC")
    st.markdown("SiB")

with row1_2:
     st.subheader('Opción 2:')
     st.metric(label="Colecciones", value="51")
     st.markdown("GrupLAC: grupos_general, investigadores, 48 Categorias más")
     st.markdown("SiB")

st.subheader('Opción 1:')
st.image(image, caption='Tamaño de la base de datos')

st.subheader('Tiempo ejecución')
row1_3, row1_4 = st.columns((2, 3))

with row1_3:
    st.subheader('Tiempo categorias GroupLAC:')
    st.metric(label="Colecciones", value="50")
    st.markdown(categorias_individual)
