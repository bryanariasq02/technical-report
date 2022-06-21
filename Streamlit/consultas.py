import pandas as pd
import numpy as np
import pymongo
import sys
import getpass
import streamlit as st
import ast 

# Referencias
# https://docs.microsoft.com/en-us/azure/cosmos-db/sql/create-cosmosdb-resources-portal
# Para ejecutar 
# python -m streamlit run prueba_base_datos.py

st.title('Consultas a la base de datos')
st.markdown('En esta seccion se pueden realizar consultas de prueba a la base de datos y la respectiva descarga.')

uri = st.text_input("Ingrese uri", key="uri")


nombre_base      = st.text_input("Ingrese nombre base de datos", key="base", value='BIOS')
nombre_coleccion = st.text_input("Ingrese nombre de la coleccion", key="col", value='SiB')
con = st.button(label='conectar')
if con:
    try: 
        client  = pymongo.MongoClient(uri)
        db_list = client.list_database_names()
        st.write('Conexion correcta! se tienen las siguientes bases de datos', db_list)
        DB      = client[nombre_base]
        col     = DB[nombre_coleccion]
    except: 
        st.write('Conexion incorrecta!')

consulta = ast.literal_eval(st.text_input("Ingrese consulta", key="consulta",  value='{}'))
button   = st.button(label='Realizar consulta')

if button: 
        st.write(consulta)
        client  = pymongo.MongoClient(uri)
        DB       = client[nombre_base]
        col      = DB[nombre_coleccion]
        query    = col.find(consulta)
        data     = [val for val in query]
        df       = convertir_df(data)
        #st.download_button(label="Download JSON", file_name="file.json", mime="application/json", data=df.to_json())
        st.download_button("Download csv", df.to_csv(),"file.csv","text/csv",key='download-csv') 
        st.write('Consulta correcta!')

#client.close()