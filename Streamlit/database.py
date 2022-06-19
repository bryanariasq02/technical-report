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
    st.markdown("GrupLAC, SiB")

with row1_2:
     st.subheader('Opción 2:')
     st.metric(label="Colecciones", value="51")
     st.markdown("GrupLAC: grupos_general, investigadores, 48 Categorias, SiB")

st.subheader('Opción 1:')
st.image(image, caption='Tamaño de la base de datos')
st.code('''from importe_datosSiB import importar_datos_SiB, buscar_datos_SiB, convertir_df 
resultados  = importar_datos_SiB()
df          = convertir_df(resultados)
df.head()''', language="python")
act = st.button(label='actualizar')

if act:
    st.write('Descargando datos!')
    resultados  = importar_datos_SiB()
    df          = convertir_df(resultados)
    st.dataframe(df.head(20))
    st.download_button("Press to Download",df.to_csv(),"file.csv","text/csv",key='download-csv')    

# Conexion con la base de datos 
st.subheader('Conexion con la base de datos')
st.markdown('A continuacion se muestra como se hace la conexion con la base de datos.')
body = '''
client  = pymongo.MongoClient(uri)
db_list = client.list_database_names()
'''
st.code(body,language="python")

# Ingreso de los datos a la base de datos 
st.subheader('Ingreso de los datos a la base de datos en mongo')
st.markdown('A continuacion se muestra como se ingresan los datos a la base de datos en mongo.')
body = '''
# Se crea la base de datos de bios 
name_DB = 'BIOS'
if name_DB not in db_list:
    DB_bios = client[name_DB]
col_list = DB_bios.list_collection_names()

# Se crea la coleccion de los datos del SiB 
col_name = 'SiB'
if col_name not in col_list: 
    SiB_col  = DB_bios[col_name]    # Coleccion de los datos del SiB
    SiB_col.insert_many(resultados) # Se insertan los datos del SiB 
'''
st.code(body,language="python")


st.title('Formas de realizar consultas a las bases de datos')
st.markdown('A continuacion se muestra como se pueden realizar consultas a la base de datos en mongo.')

body = '''
# Conexion con la base de datos
name_DB  = 'BIOS'
col_name = 'SiB'
DB_bios  = client[name_DB]
SiB_col  = DB_bios[col_name]
'''
st.code(body,language="python")

st.subheader('Primer elemento de la base de datos')
body = '''
# Retorna el primer elemento (documento) de la coleccion
query1 = SiB_col.find_one()
print('keys del resultado:', query1.keys())'''
st.code(body,language="python")

st.subheader('Elementos de toda la coleccion')
body = '''
# Retorna todos los elementos de la coleccion 
query2 = SiB_col.find({})
data   = [val for val in query2]
print('Hay ',len(data), ' elementos')
'''
st.code(body,language="python")

st.subheader('Consulta por campo')
body = '''
# Consulta por campo
myquery = {'type':'OCCURRENCE'}
query3  = SiB_col.find(myquery)
data    = [val for val in query3]
print('Hay ',len(data), ' elementos tipo OCCURRENCE')
'''
st.code(body,language="python")

st.subheader('Consulta usando expresiones regulares')
body = '''
# Consulta usando expresiones regulares
year    = '2021'
myquery = {'created': {'$regex':year}}
query4  = SiB_col.find(myquery)
data    = [val for val in query4]
print('Hay ',len(data), ' del año ', year)
'''
st.code(body,language="python")

st.subheader('ELiminar elementos de la base de datos')
body = '''
# Eliminar elementos por criterio de busqueda 
myquery = {'type':'CHECKLIST'}
query5  = SiB_col.delete_many(myquery)
query6  = SiB_col.find()
data    = [val for val in query6]
print('Hay ',len(data), ' elementos')
'''
st.code(body,language="python")

st.subheader('Actualizacion de datos')
body = '''
# Funciones auxuliares 
def list_keys(json_file, campo): 
    keys = []
    for val in json_file: 
        keys.append(val[campo])
    return(keys)
    
def add_new_files(json_file, mask): 
    new_files = []
    for val, insert in zip(json_file, mask): 
        if insert: 
            new_files.append(val)
    return(extracted_values)
# Se determina que elementos se deben agregar 
query             = SiB_col.find({})
data              = [val for val in query]
resultados_nuevos = list_keys(resultados, 'key')
resultados_viejos = list_keys(data, 'key')
mask              = [val not in resultados_viejos for val in resultados_nuevos]
agregar           = add_new_files(resultados, mask)
SiB_col.insert_many(agregar)
'''
st.code(body,language="python")

st.subheader('Orgenar datos por campo')
body = '''
# Ordenar la base de datos por titulo
SiB_col.create_index([("title",1)])
myquery = {'type':'CHECKLIST'}
query6  = SiB_col.find(myquery).limit(10).sort('title')
data    = [val['title'] for val in query6]
print('Hay ',len(data), ' elementos')
'''
st.code(body,language="python")


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