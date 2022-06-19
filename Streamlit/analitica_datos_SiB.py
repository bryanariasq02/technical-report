import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import re 
import sys
sys.path.append("C:/Users/luisalexander/Desktop/3BIO-Scientometrics/SCRIPTS/Librerias")
from importe_datosSiB import importar_datos_SiB, buscar_datos_SiB, convertir_df
import unidecode
from fuzzywuzzy import fuzz
import json
from hermetrics.levenshtein import Levenshtein
from nltk.corpus import stopwords
import nltk
#nltk.download()
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
import networkx as nx
import streamlit as st
from network_processing import generate_conections, load_results_json, create_network, draw_network
import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
# python -m streamlit run analitica_datos_SiB.py

st.title ("Aplicacion de algoritmos de analitica de datos a los datos del SiB Colombia ")
st.markdown('En este documento se presentan los resultados del analisis exploratorio de los datos captados de la API del SiB. Los datos del SiB Su principal propósito es brindar acceso abierto a información sobre la diversidad biológica del país para la construcción de una sociedad sostenible. Además, facilita la publicación en línea de datos e información sobre biodiversidad, y promueve su uso por parte de una amplia variedad de audiencias, apoyando de forma oportuna y eficiente la gestión integral de la biodiversidad.')

# Se cargan los datos 
data_SIB_merged = pd.read_csv('C:/Users/luisalexander/Desktop/3BIO-Scientometrics/DATA/DATA_RAW/SIB/data_SIB_merged.csv')
data_SIB_merged.created_datetime = pd.to_datetime(data_SIB_merged.created_datetime)
with open('C:/Users/luisalexander/Desktop/3BIO-Scientometrics/DATA/DATA_RAW/SIB/resultados_SiB.json') as json_file: resultados = json.load(json_file)

# Se definen los titulos 
st.title('Analisis exploratorio de los datos')
st.subheader('Top 10 organizaciones que mas publican datos')
st.markdown("En esta seccion se presenta un grafico con el top 10 de las organizaciones que mas publican conjuntos de datos en el SiB")

# Grafico de top 10 organizaciones 
ninst = st.slider('Numero de instituciones', 0, 50, 10)
fig, axes = plt.subplots(1,2)
Organizacion = data_SIB_merged['publishingOrganizationTitle'].value_counts()
fig = px.bar(x = Organizacion.index[0:ninst], 
       y = Organizacion.values[0:ninst], 
      title = "Top 10 organizaciones",
      labels={'x': 'Organizacion', 'y':'Frecuencia'})
st.plotly_chart(fig)

# Grafico de lenguaje usado en los datos 
st.subheader('Lenguaje usado en la publicacion de los datos')
st.markdown("En esta seccion se presenta un grafico con el porcentaje de uso del lengua en el que se publicaron los datos")
lenguaje      = data_SIB_merged['language'].value_counts()
fig2 = px.pie(values = lenguaje.values, names = lenguaje.index, title='Lenguaje')
st.plotly_chart(fig2)

# Cantidad de conjuntos agregados por año
st.subheader('Cantidad de datos publicados por año')
st.markdown("En esta seccion se presenta un grafico con la cantidad de conjuntos de datos publicados cada año")
cantidad_ano = data_SIB_merged.created_datetime.dt.year.value_counts()
fig3         = px.area(x = cantidad_ano.index, y=cantidad_ano.values, title='Publicaciones por año')
fig3.update_xaxes(title_text='Año')
fig3.update_yaxes(title_text='Cantidad')
st.plotly_chart(fig3)

tasa_cambio =cantidad_ano.sort_index().pct_change()*100
fig31= px.area(x = tasa_cambio.index, y = tasa_cambio.values, title='Porcentaje de cambio en cantidad de publicaciones por año')
st.plotly_chart(fig31)

# Hosting organization 
st.subheader('Organizacion anfitriona')
st.markdown("En esta seccion se presenta un grafico con la cantidad de conjuntos de datos almacenados en cada organizacion anfitriona")
hostingOrganization = data_SIB_merged.hostingOrganizationTitle.value_counts()
fig4 = px.bar(x = hostingOrganization.index[0:10], 
             y = hostingOrganization.values[0:10], 
             title = "Hosting organization",
             labels={'x': 'Hosting Organization', 'y':'Frecuencia'})
st.plotly_chart(fig4)      


# Tipo de conjunto de datos  
st.subheader('Tipo de conjunto de datos')
st.markdown('En esta seccion se presenta un grafico con el porcentaje de tipos de datos almacenados en la base de datos del SiB')
tipo = data_SIB_merged['type'].value_counts()
fig5 = px.pie(values = tipo.values, names = tipo.index, title='Tipo de conjunto de datos')
st.plotly_chart(fig5)  

# Analisis del lenguaje en los titulos 
st.title('Procesamiento del lenguaje natural')
st.markdown('En esta seccion se presenta un analisis mediante procesamiento del lenguaje natural de los titulos y descripcion de los conjuntos de datos.')
st.subheader('Bi-grams mas frecuentes en los titulos')
st.markdown("En esta seccion se presenta un grafico con el top 10 de los Bi-grams mas frecuentes en los titulos de los conjuntos de datos")

stoplist     = stopwords.words('spanish')
c_vec        = CountVectorizer(stop_words = stoplist, ngram_range=(2,2))
ngrams       = c_vec.fit_transform(data_SIB_merged['title'])
count_values = ngrams.toarray().sum(axis=0)
vocab        = c_vec.vocabulary_
df_ngram = pd.DataFrame(sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)
            ).rename(columns={0: 'frequency', 1:'n-gram'})

fig6 = px.bar(df_ngram.head(10), x = 'n-gram', 
       y = 'frequency', 
      title = "Analisis de n-grams",
      labels={'x': 'n-gram', 'y':'Frecuencia'})
st.plotly_chart(fig6) 

# En la distribucion de la descripcion 
st.subheader('Bi-grams mas frecuentes en la descripcion')
st.markdown("En esta seccion se presenta un grafico con el top 10 de los Bi-grams mas frecuentes en la descripcion de los conjuntos de datos")

c_vec        = CountVectorizer(stop_words = stoplist, ngram_range=(2,2))
ngrams       = c_vec.fit_transform(data_SIB_merged['description'])
count_values = ngrams.toarray().sum(axis=0)
vocab        = c_vec.vocabulary_
df_ngram = pd.DataFrame(sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)
            ).rename(columns={0: 'frequency', 1:'n-gram'})

fig7 = px.bar(df_ngram.head(10), x = 'n-gram', 
       y = 'frequency', 
      title = "Analisis de n-grams",
      labels={'x': 'n-gram', 'y':'Frecuencia'})
st.plotly_chart(fig7) 

# En la distribucion de la descripcion 
st.subheader('Worcloud de palabras en la descripcion')
st.markdown("En esta seccion se presenta un worcloud con las palabras de las descripciones de los conjuntos de datos")
word_cloud_text = ' '.join(data_SIB_merged.description).replace('<p','').replace('p>','')
wordcloud = WordCloud(max_font_size=100, max_words=100, background_color="white",\
                       scale = 10,width=800, height=400, stopwords = stoplist).generate(word_cloud_text)
fig, ax = plt.subplots(figsize=(15,10))
ax.imshow(wordcloud, interpolation="bilinear")
ax.axis("off")
st.pyplot(fig)

# Analisis de relaciones entre instituciones 
st.title('Analisis de relaciones entre organizaciones, paises y ciudades')
st.markdown('En esta seccion se presenta un analisis de relaciones entre las organizaciones, paises e instituciones que publican o han publicado conjuntos de datos en el SiB')

file_path  = 'C:/Users/luisalexander/Desktop/3BIO-Scientometrics/DATA/DATA_RAW/SIB/resultados_SiB.json'
resultados = load_results_json(file_path)
conections = generate_conections(resultados)

# ---------------- RELACIONES ENTRE ORGANIZACIONES ----------------------------------------
st.subheader('Relacion entre organizaciones y cantidad de conexiones por año')
st.markdown("En esta seccion se presenta un grafo con las conexiones entre organizaciones, el grafico de cantidad de conexiones por año, un grafico de cantidad de conexiones nuevas cada año y una tabla de las organizaciones con la mayor cantidad de relaciones")
network  = create_network(conections['organizacion'])
G        = network['Graph']
position = nx.fruchterman_reingold_layout(G)
fig      = draw_network(G, position, node_size = 10, title='Relaciones entre organizaciones')
st.plotly_chart(fig) 
st.write('Nodos mas influyentes')
st.write(nx.voterank(G, number_of_nodes=10))


number_conections = []
node_number       = []
years = pd.to_datetime(conections['year']).year.value_counts().sort_index().index
mask  = 0
for year in years:
    mask  = mask | pd.Series(conections['year']).str.contains(str(year))
    val   = create_network(pd.Series(conections['organizacion'])[mask].values)
    G     = val['Graph']
    number_conections.append(G.number_of_edges())
    node_number.append(len(G.nodes))

fig = px.area(x=years, y= number_conections, title='Numero de conexiones por año')
st.plotly_chart(fig) 

fig = go.Figure()
fig.add_trace(go.Scatter(x = years, y = number_conections, fill='tozeroy', name = 'Numero de conexiones')) # fill down to xaxis
fig.add_trace(go.Scatter(x = years, y = node_number, fill='tonexty', name = 'Numero de nodos')) # fill to trace0 y
fig.update_layout(title="Numero de nodos vs Numero de conexiones", showlegend=True)
st.plotly_chart(fig) 

fig = go.Figure()
fig.add_trace(go.Scatter(x = years, y = number_conections, fill='tozeroy',  stackgroup='one',groupnorm='percent', name = 'Numero de conexiones')) # fill down to xaxis
fig.add_trace(go.Scatter(x = years, y = node_number, fill='tonexty', groupnorm='percent', stackgroup='one', name = 'Numero de nodos')) # fill to trace0 y
fig.update_layout(title="Numero de nodos vs Numero de conexiones normalizado", showlegend=True, xaxis_type='category', yaxis=dict(type='linear', range=[1, 100], ticksuffix='%'))
st.plotly_chart(fig) 


number_conections.insert(0,0)
cambio_year = np.diff(number_conections)
fig = px.area(x=years, y= cambio_year, title='Conexiones nuevas por año', labels={'x': 'Año', 'y':'Cantidad'})
st.plotly_chart(fig) 


org   = []
grado = []
for organ, degree in G.degree:
    org.append(organ)
    grado.append(degree)
nodes_degree = pd.DataFrame({'Organizacion':org, 'Degree':grado})

st.dataframe(nodes_degree.sort_values(by='Degree',ascending=False).head(10))


# ---------------- RELACIONES ENTRE CIUDADES ----------------------------------------
st.title('Analisis de relaciones ciudades y cantidad de conexiones por año')
st.markdown('En esta seccion se presenta un grafo con las conexiones entre ciudades, el grafico de cantidad de conexiones por año, un grafico de cantidad de conexiones nuevas cada año y una tabla de las ciudades con la mayor cantidad de relaciones')
network  = create_network(conections['ciudad'])
G        = network['Graph']
position = nx.fruchterman_reingold_layout(G)
fig      = draw_network(G, position, node_size = 10, title='Relaciones entre ciudades')
st.plotly_chart(fig) 
st.write('Nodos mas influyentes')
st.write(nx.voterank(G, number_of_nodes=10))


number_conections = []
node_number       = []
years = pd.to_datetime(conections['year']).year.value_counts().sort_index().index
mask  = 0
for year in years:
    mask  = mask | pd.Series(conections['year']).str.contains(str(year))
    val   = create_network(pd.Series(conections['ciudad'])[mask].values)
    G     = val['Graph']
    number_conections.append(G.number_of_edges())
    node_number.append(len(G.nodes))
fig = px.area(x=years, y= number_conections, title='Numero de conexiones por año')
st.plotly_chart(fig) 

fig = go.Figure()
fig.add_trace(go.Scatter(x = years, y = number_conections, fill='tozeroy', name = 'Numero de conexiones')) # fill down to xaxis
fig.add_trace(go.Scatter(x = years, y = node_number, fill='tonexty', name = 'Numero de nodos')) # fill to trace0 y
fig.update_layout(title="Numero de nodos vs Numero de conexiones", showlegend=True)
st.plotly_chart(fig) 

fig = go.Figure()
fig.add_trace(go.Scatter(x = years, y = number_conections, fill='tozeroy',  stackgroup='one',groupnorm='percent', name = 'Numero de conexiones')) # fill down to xaxis
fig.add_trace(go.Scatter(x = years, y = node_number, fill='tonexty', groupnorm='percent', stackgroup='one', name = 'Numero de nodos')) # fill to trace0 y
fig.update_layout(title="Numero de nodos vs Numero de conexiones normalizado", showlegend=True, xaxis_type='category', yaxis=dict(type='linear', range=[1, 100], ticksuffix='%'))
st.plotly_chart(fig) 

number_conections.insert(0,0)
cambio_year = np.diff(number_conections)
fig = px.area(x=years, y= cambio_year, title='Conexiones nuevas por año', labels={'x': 'Año', 'y':'Cantidad'})
st.plotly_chart(fig) 


ciu   = []
grado = []
for val, degree in G.degree:
    ciu.append(val)
    grado.append(degree)
nodes_degree = pd.DataFrame({'Ciudad':ciu, 'Degree':grado})

st.dataframe(nodes_degree.sort_values(by='Degree',ascending=False).head(10))

# ---------------- RELACIONES ENTRE PAISES ----------------------------------------
st.title('Analisis de relaciones paises y cantidad de conexiones por año')
st.markdown('En esta seccion se presenta un grafo con las conexiones entre paises, el grafico de cantidad de conexiones por año, un grafico de cantidad de conexiones nuevas cada año y una tabla de las instituciones con la mayor cantidad de relaciones')
network  = create_network(conections['country'])
G        = network['Graph']
country_codes      = pd.read_csv('C:/Users/luisalexander/Desktop/3BIO-Scientometrics/DATA/DATA_RAW/SIB/world.csv')
reemplacement_code = {alpha2:country for alpha2, country in zip(country_codes['alpha2'],country_codes['es'])}
G = nx.relabel_nodes(G, reemplacement_code)
position = nx.fruchterman_reingold_layout(G)
fig      = draw_network(G, position, node_size = 10,  title='Relaciones entre paises')
st.plotly_chart(fig) 
st.write('Nodos mas influyentes')
st.write(nx.voterank(G, number_of_nodes=10))


number_conections = []
node_number       = []
years = pd.to_datetime(conections['year']).year.value_counts().sort_index().index
mask  = 0
for year in years:
    mask  = mask | pd.Series(conections['year']).str.contains(str(year))
    val   = create_network(pd.Series(conections['country'])[mask].values)
    G     = val['Graph']
    number_conections.append(G.number_of_edges())
    node_number.append(len(G.nodes))
fig = px.area(x=years, y= number_conections, title='Numero de conexiones por año')
st.plotly_chart(fig) 


fig = go.Figure()
fig.add_trace(go.Scatter(x = years, y = number_conections, fill='tozeroy', name = 'Numero de conexiones')) # fill down to xaxis
fig.add_trace(go.Scatter(x = years, y = node_number, fill='tonexty', name = 'Numero de nodos')) # fill to trace0 y
fig.update_layout(title="Numero de nodos vs Numero de conexiones", showlegend=True)
st.plotly_chart(fig) 

fig = go.Figure()
fig.add_trace(go.Scatter(x = years, y = number_conections, fill='tozeroy',  stackgroup='one',groupnorm='percent', name = 'Numero de conexiones')) # fill down to xaxis
fig.add_trace(go.Scatter(x = years, y = node_number, fill='tonexty', groupnorm='percent', stackgroup='one', name = 'Numero de nodos')) # fill to trace0 y
fig.update_layout(title="Numero de nodos vs Numero de conexiones normalizado", showlegend=True, xaxis_type='category', yaxis=dict(type='linear', range=[1, 100], ticksuffix='%'))
st.plotly_chart(fig) 


number_conections.insert(0,0)
cambio_year = np.diff(number_conections)
fig = px.area(x=years, y= cambio_year, title='Conexiones nuevas por año', labels={'x': 'Año', 'y':'Cantidad'})
st.plotly_chart(fig) 

G = nx.relabel_nodes(G, reemplacement_code)
pa    = []
grado = []
for val, degree in G.degree:
    pa.append(val)
    grado.append(degree)
nodes_degree = pd.DataFrame({'Pais':pa, 'Degree':grado})

st.dataframe(nodes_degree.sort_values(by='Degree',ascending=False).head(10))

