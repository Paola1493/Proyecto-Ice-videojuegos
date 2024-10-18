
# # Proyecto integrado
# 
# La tienda online Ice que vende videojuegos por todo el mundo. Las reseñas de usuarios y expertos, los géneros, las plataformas (por ejemplo, Xbox o PlayStation) y los datos históricos sobre las ventas de juegos están disponibles en fuentes abiertas. Identificaremos patrones que determinen si un juego tiene éxito o no. Esto te permitirá detectar proyectos prometedores y planificar campañas publicitarias.


# # Paso 1. Abre el archivo de datos y estudia la información general 

# In[38]:


import pandas as pd
import seaborn as sns 
import numpy as np 
from math import factorial
from scipy import stats as st
import math as mt
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind


# In[2]:


data_games= pd.read_csv('/datasets/games.csv')


# In[3]:


display(data_games.head())
display(data_games.info())


# # Paso 2. Prepara los datos

# In[4]:


data_games.columns =data_games.columns.str.lower()
data_games.head()


# In[5]:


data_games.info()


# In[6]:


display(data_games[data_games['name'].isna()])

# # Comentarios
# 
# Procedí a eliminar dos filas del que tenían valores NaN en 5 de 11 columnas del dataset. Sus ventas ventas eran bajas, por lo tanto, sacarlos no influye en el análisis de datos posterior.
 

# In[8]:


display(data_games.duplicated().sum())


# In[9]:


serie_imputacion = data_games.groupby('name')['year_of_release'].median()
data_games['year_of_release']= data_games.set_index('name')['year_of_release'].fillna(serie_imputacion).reset_index(drop=True)
data_games.head()


# # Comentarios
# 
# Se reemplazaron los valores ausentes de la columna 'year_of_release' para poder tener información más clara al momento de generar los gráficos que necesitaran la información de los años de lanzamiento de los juegos. Esto se realizó a través de la imputación de los años de lanzamiento correspondientes a cada juego y la información se extrajo de las otras plataformas en las que el juego fue lanzado.
# 
# ¿Por qué crees que los valores están ausentes? Brinda explicaciones posibles.
# 
# Los valores ausente podrían deberse a que en años anteriores no se llevaba, como hoy en día, un registro tan específico de los juegos que se lanzaban. Ta,mbién puede deberse a que los juegos fueron lanzados en diferentes plataformas en mismo año, por lo tanto no se traspasó la información a las otras plataformas de la data.

# In[10]:


display(data_games[data_games['year_of_release'].isna()])


# In[11]:


data_games= data_games.dropna(subset=['year_of_release'])
data_games.info()


# # Comentarios
# 
# Se eliminaron los valores ausentes de la columna 'year_of_release' que por algún motivo no pudieron ser reemplazados. Sólo eran 4 filas, por lo que los resultados del análisi no debieran verse alterados.

# In[12]:


data_games['year_of_release'] = data_games['year_of_release'].astype('int')
display(data_games)


# # Comentarios
# 
# se cambió el tipo de dato float de la columna 'year_of_release' a int

# In[13]:


data_games['user_score']= pd.to_numeric(data_games['user_score'], errors= 'coerce')


# # Comentarios
# 
# Los valores tbd fueron reemplazados por nan para que al momento de hacer algún cálculo con esta columna no se generaran errores.

# In[14]:


data_games['total_sales']= data_games['na_sales']+data_games['eu_sales']+data_games['jp_sales']+data_games['other_sales']
data_games.head()


# # Comentarios
# 
# Se creó una nueva columna llamada total_sales para tener la sumatoria del total de las ventas de cada juego en todas las regiones.


# # Paso 3. Analiza los datos

# In[46]:


games_per_year= data_games.groupby('year_of_release')['name'].count().reset_index()
games_per_year.plot(kind= 'bar', title= 'Juegos lanzados por año', x='year_of_release', y= 'name', figsize= [10, 6])


# # Comentarios
# 
# En este gráfico se puede ver el crecimiento que existió en el lanzamiento de juegos a partir del año 1995, alcanzando su pick en el 2008 y manteniendose en el 2009.

# In[49]:


games_per_platform= data_games.groupby('platform')['total_sales'].sum().reset_index()
games_per_platform.plot(kind= 'bar', title= 'Ventas por plataforma', x='platform', y= 'total_sales',figsize= [10, 6])


# # Comentarios
# 
# En este gráfico se puede observar que las 5 plataformas más populares son DS, X360, PS3, Wii y PS2. Por otro lado, las menos populares son 3DO, GG, NG, PCFX, SCD, TG16 y WS.

# In[17]:


platform_list= ('PS2', 'X360', 'PS3', 'Wii', 'DS')
data_games_2= data_games[data_games['platform'].isin(platform_list)]


# In[18]:


platform_distribition= data_games_2.groupby(['year_of_release', 'platform'])['total_sales'].sum().reset_index()


# In[19]:


fig, ax = plt.subplots(figsize = (12, 8))
sns.lineplot(data= platform_distribition, x= 'year_of_release', y= 'total_sales', hue= 'platform', ax= ax)


# # Comentarios
# 
# 
# Por lo que muestra el gráfico, el pik de ventas de una plataforma es de aproximadamente 5 años y luego las ventas comienzan a decaer.
#  
#  Las nuevas plataformas aparecen, aproximadamente, a los 2.5 años desde que se lanzó la anterior.

# In[20]:


data_games_3= data_games_2[data_games_2['year_of_release'] > 2010].reset_index()
data_games_3


# In[21]:


sns.boxplot(data= data_games_3, x= 'total_sales', y='platform', showfliers= False)


# # Comentarios
# Las diferencias de las ventas del período de 5 años elegido son significativas, mostrando que PS3 es la consola más popular y DS la menos popular. Por otro lado, estos diagramas de caja muestran que, una vez que las consolas alcanzan su peak en popularidad este disminuye y los datos del Q3 en todas las cajas son más dispersos.

# In[22]:


games_score= data_games_3.groupby(['critic_score', 'user_score'])['total_sales'].sum().reset_index()
games_score


# In[56]:


fig, ax = plt.subplots(figsize = (10, 8))
sns.scatterplot(x = "critic_score", y = "total_sales", data = games_score, ax= ax)


# In[54]:


fig, ax = plt.subplots(figsize = (10, 8))
sns.scatterplot(x = "user_score", y = "total_sales", data = games_score, ax= ax)


# In[71]:


corr= games_score.corr()
corr


# # Comentarios
# 
# 1. correlación critic_score/total_sales: 0.428834
# 2. correlación user_score/total_sales: 0.037212
# 
# Según los datos que se pueden observar en los gráficos de dispersión y en el cálculo de la correlación entre las variables, no existe una relación entre éstas en ninguno de los dos casos.


# In[23]:


games_per_genre = data_games_3.groupby('genre')['total_sales'].sum().reset_index()
games_per_genre.plot(kind= 'bar', x='genre', y= 'total_sales', figsize= [10, 6])


# # Comentarios
# 
# 
# Los géneros más rentables son el de accion, disparos y desportes. 
# Por lo que muestra el gráfico los géneros más rentables son los que tienen juegos con mayor actividad y entretenimiento es éste, miestras que los menos rentables, por lo general, son juegos de estrategia, puzzle, etc, y éstos sueles ser menos dinámicos.

# # Paso 4. Crea un perfil de usuario para cada región
# 

# In[24]:


market= data_games.groupby('platform').agg({'na_sales': 'sum',
                                   'eu_sales': 'sum',
                                   'jp_sales': 'sum'
                                    }).reset_index()
market


# In[25]:


market_sales= pd.melt(market, id_vars= ['platform'], value_vars= ['na_sales', 'eu_sales', 'jp_sales'], var_name= 'market', value_name= 'sales')
market_sales


# In[68]:


fig, ax = plt.subplots(figsize = (15, 10))
sns.barplot(data= market_sales, x= 'market', y= 'sales', hue= 'platform', ax= ax)


# # Comentarios
# 
# 
# En este gráfico se pueden apreciar las consolas más populares por región, estas son: 
# 
# 'na_sales': X360, PS2, Wii, PS3 y DS.
# 
# 'eu_sales': PS2, PS3, X360, Wii y PS.
# 
# 'jp_sales': DS, PS, PS2, SNES y 3DS.
# 
# La variación de las cuotas de mercado por región las lidera na_sales, quien obtiene los mayores ingresos, luego se ecuentra eu_sales y finalmente, con las menores cuotas está jp_sales.

# In[27]:


market_genre= data_games.groupby('genre').agg({'na_sales': 'sum',
                                   'eu_sales': 'sum',
                                   'jp_sales': 'sum'
                                    }).reset_index()
market_genre


# In[28]:


market_sales_genre= pd.melt(market_genre, id_vars= ['genre'], value_vars= ['na_sales', 'eu_sales', 'jp_sales'], var_name= 'market', value_name= 'sales')
market_sales_genre


# In[29]:


fig, ax = plt.subplots(figsize = (12, 10))
sns.barplot(data= market_sales_genre, x= 'market', y= 'sales', hue= 'genre', ax= ax)


# # Comentarios
# 
# En este gráfico se pueden apreciar los géneros de juegos más populares por región, estos son: 
# 
# 'na_sales': Action, Sports, Shooter, Platform y Misc.
# 
# 'eu_sales': Action, Sports, Shooter, Racing y Misc.
# 
# 'jp_sales': Rol-playing, Action, Sports, Platform y Misc.
# 
# 

# In[41]:


score_xone= data_games[data_games['platform']== 'XOne']['user_score'].dropna()
score_pc= data_games[data_games['platform']== 'PC']['user_score'].dropna()


# In[42]:


#PRUEBA DE HIPÓTESIS
# H0: Las calificaciones promedio de los usuarios para las plataformas Xbox One y PC son las mismas.
# H1: Las calificaciones promedio de los usuarios para las plataformas Xbox One y PC diferentes.

alpha = 0.05 
stat, p_value= ttest_ind(
    score_xone, 
    score_pc
)

print(f"""
    La calificación promedio de los usuarios para XOne es: {score_xone.mean()}
    La calificación promedio de los usuarios para XOne es: {score_pc.mean()}
    
    t-statistic: {stat}
    p-value: {p_value}
""")


if p_value < alpha: 
    print("Rechazamos la hipótesis nula")
else:
    print("No podemos rechazar la hipótesis nula") 


# # Comentarios
# 
# 
# El resultado de la prueba arroja que el valor p es 1.26813103168632e-05, por lo tanto se debe rechazar la hipótesis nula, es decir, que si existe diferencia entre las calificaciones promedio de los usuarios para las plataformas Xbox One y PC.

# In[43]:


score_action= data_games[data_games['genre']== 'Action']['user_score'].dropna()
score_sports= data_games[data_games['genre']== 'Sports']['user_score'].dropna()


# In[44]:


#H1: Las calificaciones promedio de los usuarios para los géneros de Acción y Deportes son las mismas.
#H1: Las calificaciones promedio de los usuarios para los géneros de Acción y Deportes son diferentes.

alpha = 0.05 
stat, p_value= ttest_ind(
    score_action, 
    score_sports
)

print(f"""
    La calificación promedio de los usuarios para el género de Acción es: {score_action.mean()}
    La calificación promedio de los usuarios para el género de Sports es: {score_sports.mean()}
    
    t-statistic: {stat}
    p-value: {p_value}
""")


if p_value < alpha: 
    print("Rechazamos la hipótesis nula")
else:
    print("No podemos rechazar la hipótesis nula") 


# # Comentarios
# El resultado de la prueba arroja que el valor p es 0.07346036608929722, por lo tanto no se debe rechazar la hipótesis nula, es decir, que las calificaciones promedio de los usuarios para los géneros de Acción y Deportes son las mismas.

# # CONCLUSIONES
# 

# 1. El éxito de los lanzamientos de consolas comenzó a partir de 1995.
# 2. Las 5 plataformas más populares son DS, X360, Wii, PS3 y PS2.
# 3. El éxito de las consolas, por lo general, dura 5 años y a partir de la mitad de este tiempo surgen nuevas consolas.
# 4. Respecto a los datos de 2010 en adelante:
#     a) No existe una correlación entre los user/critic_ score y las ventas totales en cada plataforma.
#     b) Los géneros más populares son Action, Shooter y Sports.
# 5. Según las pruebas de hipótesis: 
#     a) Existe diferencia entre las calificaciones promedio de los usuarios para las plataformas Xbox One y PC.
#     b) Las calificaciones promedio de los usuarios para los géneros de Acción y Deportes son las mismas.
#
# <br>Es un placer reconocer tu dedicación y el análisis detallado que has llevado a cabo. Continúa superándote en tus futuras iniciativas. Confío en que aplicarás este conocimiento de manera efectiva en desafíos futuros, avanzando hacia objetivos aún más ambiciosos.
# </div>
# 
