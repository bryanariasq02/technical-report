import numpy as np
import pandas as pd

#datos_tiempo = pd.DataFrame()
#datos_tiempo = datos_tiempo.append({'Lista subida individual' : tiempo_GrupLAC_subida}, ignore_index=True)

## Datos GrupLAC en minutos
promedio_categoria = 142.094/60
promedio_categoria = np.around(promedio_categoria, 2)
suma_subida_categorias = 7068.66/60
suma_subida_categorias = np.around(suma_subida_categorias, 2)
tiempo_general_GrupLAC = 7119.19/60
tiempo_general_GrupLAC = np.around(tiempo_general_GrupLAC, 2)
#dataframes
categorias_GrupLAC = pd.DataFrame()
#datos_tiempo = datos_tiempo.append({'Lista subida individual' : tiempo_GrupLAC_subida}, ignore_index=True)

## Datos SiB en segundos
tiempo_SiB_general = 16.4923355557937622
tiempo_SiB_general = np.around(tiempo_SiB_general, 2)
tiempo_SiB_subida = 70.57624459266663
tiempo_SiB_subida = np.around(tiempo_SiB_subida, 2)
total_tiempo_SiB = tiempo_SiB_general + tiempo_SiB_subida

## Tiempo general en horas
tiempo_total = 7209.841993093491/3600
tiempo_total = np.around(tiempo_total, 4)
