#!/usr/bin/env python
# coding: utf-8

# # PRÁCTICA Machine Learning
# 
# Tenéis que predecir el precio de un airbnb utilizando los datos disponibles. 
# Se valorará: 
#    * Generación de nuevas características a partir de las existentes 
#    * Análisis exploratorio 
#    * Selección y evaluación de modelo 
#    * Comparativa de distintos algoritmos 

# ### 0. Importar Librerías

# In[2]:


# Importación de librerías
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
get_ipython().run_line_magic('matplotlib', 'inline')

#color map
cm = plt.cm.RdBu  
cm_bright = ListedColormap(['#FF0000', '#0000FF'])

import warnings
warnings.filterwarnings('ignore')


# ### 1. Cargar los datos del Airbnb

# In[1052]:


# Una vez que hemos descargado el dataset y guardado en la carpeta data, cargamoos los datos en un dataframe.
airbnb = pd.read_csv('./data/airbnb-listings-extract.csv', sep=';' , decimal='.')


# In[5]:


airbnb.head()
pd.set_option('display.max_rows', 89)
airbnb.iloc[50, :]
#airbnb.head(89)


# ### 2. Análisis exploratorio
# 
# #### 2.1 Una vez cargado el data set, vamos a realizar un análisis básico, donde vamos a analizar las columnas. 
# 
# Con el comando airbnb.columns veo que el dataframe tiene **89** columnas. He eliminado:
# - Todas las columnas que contienen URLs.
# - Las columnas de *Scrape ID, Last Scraped, Calendar last Scraped y ID* porque son datos que nos indican cuando se accedió a la pagina de airbnb para obtener todos los datos de cada airbnb y esto no va a afectar al precio del airbnb. 
# - Columnas de descripcion del airbnb como *Name, Summary , Space, Description, Experiences Offered, Neighborhood Overview, Notes, Transit, Access, Interaction, House Rules, Host Name, Host About*, de las que no puedo sacar información válida sin procesar antes con NLP.
# 
# Después de eliminar estas columnas, me quedo con un total de **63** columnas. 
# 
# Tengo dudas sobre las siguientes columnas: 
# - *Weekly price* vs *Monthly price* vs *Price*
# - *Guest included* vs *Extra people*
# - *Latitude* y *Longitud* vs *Geolocation*
# - *City* vs *Market* vs *Smart location*
# - *Country code* vs *Country*
# - *Neighbourhood* vs *Neighbourhood Cleansed* vs *Neighbourhood Group Cleansed*
# - *Host Listing Count* vs *Host Total Listing Count* vs *Calculated host listing count*
# - *Bed Type*
# - Fechas : *Calendar updated* , *first review* and *last review*
# - *Has Availability, Availability 30, Availability 60, Availability 90, Availability 365*    
# - *License*
# - *Jurisdiction Names*
# - *Features*
# 
# **NOTAS:** <br>
# Para calcular el precio, veo que existe la columna de *Price*, pero también la de *Security deposit* y *Cleaning fee*, de forma que el precio total será la suma de: price + cleaning fee siendo security deposit optativo y no siempre necesario.
# 
# Significado de algunas columnas: <br>
# *Guest included* - numero de personas incluidas en el precio anunciado<br>
# *Extra people* - Precio a añadir por cada extra persona al numero de "guest included"<br>
# *Availability 365* - El número de dias que un particular host está disponible en 365 dias (1 año)

# In[ ]:


# Analizar columnas y eliminarlas
airbnb.columns

to_drop = ['Listing Url','Thumbnail Url','Medium Url','Picture Url', 'XL Picture Url', 'Host URL', 'Host Thumbnail Url',
           'Host Picture Url', 'ID', 'Scrape ID','Host ID', 'Last Scraped','Calendar last Scraped' , 'Name', 'Summary' ,
           'Space' , 'Description' , 'Experiences Offered', 'Neighborhood Overview' , 'Notes' , 'Transit' , 'Access', 
           'Interaction', 'House Rules' , 'Host Name' , 'Host About']
 

#airbnb.drop(to_drop, inplace=True, axis=1)
len (airbnb.columns)


# #### 2.2 Filtrado por Ciudad de Madrid
# Una vez eliminado las columnas que no nos aportan información para el precio, debido a que conozco este data set de las prácticas anteriores, sabemos que aunque el fichero se ha hecho filtrando la ciudad de Madrid, no todos los registros son de Madrid, por lo que antes de hacer la separacion a train/test, voy a filtrar para quedarme con los registros en los que la columna city es únicamente **Madrid.**

# In[ ]:


# 1572 registros han sido eliminados
filter_Madrid = airbnb['City'] == 'Madrid'
airbnb = airbnb[filter_Madrid]
#airbnb.head()


# #### 2.3 Realizar la separacion de train y test
# Vamos a realizar la separacion del data set en el grupo training y grupo test. Para ello vamos a guardar los datos en dos ficheros diferentes .csv para que sea más facil trabajar con ellos
# Es muy importante obtener un modelo basándonos **solo** en los datos de training. Una vez conseguido el modelo que puede generalizar mejor la solución al problema que buscamos, probamos ese modelo en el grupo de test para evaluar nuestro modelo. 
# 
# Lo que se busca es encontrar un modelo que funcione bien en datos que no han sido vistos antes (test data set). 

# In[ ]:


from sklearn.model_selection import train_test_split

# 20% de los datos para test y 80% de los datos para training
train, test = train_test_split(airbnb, test_size=0.2, shuffle=True, random_state=0)

print(f'Dimensiones del dataset de training: {train.shape}')
print(f'Dimensiones del dataset de test: {test.shape}')

# Guardamos los dos grupos en ficheros csv ya que será mas facil realizar el preprocesado en ellos con pandas
train.to_csv('./data/train.csv', sep=';', decimal='.', index=False)
test.to_csv('./data/test.csv', sep=';', decimal='.', index=False)


# In[1302]:


# Cargamos el dataset de train y trabajamos ÚNICAMENTE con él para hacer el estudio exploratorio
airbnb_train = pd.read_csv('./data/train.csv', sep=';', decimal='.')


# #### 2.4 Análisis individual de cada caracteristica. 

# In[1303]:


# Evaluamos el tipo de cada columna
#airbnb_train.dtypes


# Encontramos que hay una característica *Square Feet* dimensión en pies al cuadrado y la convertimos en metros cuadrados, *Square Meters*.

# In[1304]:


# Convertir la variable Square Feet a metros cuadrados y cambiar el nombre
airbnb_train['Square Feet'] = airbnb_train['Square Feet']* 0.3048 * 0.3048

# Renombramos
airbnb_train.rename(columns={'Square Feet': 'Square Meters'}, inplace=True)


# Como Nuestro estudio esta centrado en la ciudad de Madrid, puedo quitar las siguientes variables, que hacen referencia a la localización y no aportan información adicional.
# 
#  - *Street, City, State, Market, Smart Location, Country Code , Country,  Zipcode*.
#  - También la variable *Geolocation*, está compuesta por las dos variables *Latitud* y *Longitud*.
#  
# NOTA : Hemos usado  ```airbnb_train['variable'].value_counts() ``` para ver y evaluar el contenido de estas columnas. 

# In[1305]:


to_drop_location = ['Street', 'City', 'State', 'Market', 'Smart Location' , 'Country Code', 'Country' , 
                    'Zipcode' , 'Geolocation']
airbnb_train.drop(to_drop_location, inplace=True, axis=1)


# Antes de hacer cualquier conversión categórica por columna, voy a contar el número de registros vacíos por columna.  
# Sabiendo que nuestro airbnb_train contiene 10565 registros, y usando el comando ```airbnb_train.isna().sum(axis=0)``` vemos la cantidad de valores vacios para cada columna:    
# - *Host Acceptance Rate*: 10565 , *Has Availabity*: 10565 , *Jurisdiction Names* : 10565 
# - *License* : 10379
# - *Square Meters* : 10138 
# - *Weekly Price*: 7889 y *Monthly Price*: 7931   
# 
# Para las características de la primera fila, ya que no hay valores para ningun registro, las eliminamoos directamente.   
# Para la característica de *License* no tenemos datos para un 98% de los registros
# En el caso de *Square Meters*, aunque parece una caracteristica relevante, no hay suficiente datos para ofrecer información relevante y si imputamos, practicamente todos los valores van a ser iguales, lo que no va a proporcionar información adicional al modelo.
# Y para el último caso, la característica *Price* solo tiene 7 registros sin valores, mientras que *Weekly Price* y *Monthly Price* nos aportan la misma información basada en diferente criterio, y ademas tienen alrededor del 75% de registros vacíos. 
# 
# Por estas razones, vamos a proceder a eliminar también estas caracteristicas de nuestro estudio, quedándonos con **42** columnas para cada registro.

# In[1306]:


airbnb_train.isna().sum(axis=0)


# In[1307]:


airbnb_train.drop(['Host Acceptance Rate', 'Square Meters', 'Has Availability', 'Jurisdiction Names',
                   'License', 'Weekly Price' , 'Monthly Price' ], inplace=True, axis=1)


# In[1308]:


# Setting para mostrar todas las filas
pd.set_option('display.max_rows', None)

object_columns = len(airbnb_train.select_dtypes('object').columns)
print (f'El número de caracteristicas tipo object son: {object_columns} ')
airbnb_train.select_dtypes('object').columns


# Analizando cada una de estas caracteristicas, usando ```airbnb_train['caracteristica'].value_counts()``` y ```airbnb_train['caracteristica'].unique()``` llegamos a la siguiente conclusión:
# 
# - *Host Location* - tiene 8205 registros en Madrid. El resto corresponde a 436 diferentes localizaciones. El contenido tiene mucho ruido, ya que a veces esta solo la ciudad, o solo el pais, o una direccion de calle sin ciudad ni pais, e incluso descripciones, consideramos que este valor no es importante para evaluar el precio del airbnb
# - *Host Neighbourhood* - es la misma información que Host Location. La mayoria de Madrid, 2620 NAN y unos 150 registros que no son de Madrid que corresponden a unos 40 barrios diferentes.
# - *Calendar Updated*, fechas de los 3 últimos años y 66 registros nomeados como "nunca" ; *First Review* 2187 registros NAN ; *Last Review* 2188 registros NAN - son simplemente fechas que no nos aportan información sobre el precio del airbnb.
# 
# Por estas razones vamos a eliminar estas columnas también.    
# Por lo que al final nos quedamos con 17 - 5 = **12** características object que es necesario convertir y codificar en algunos casos.

# In[1309]:


airbnb_train.drop(['Host Location', 'Host Neighbourhood', 'Calendar Updated',
                   'First Review' ,'Last Review'], inplace=True, axis=1)


# In[1310]:


airbnb_train.iloc[50, :]
len(airbnb_train.columns)


# In[1311]:


airbnb_train.describe().T


# In[1312]:


# Pintamos histogramas para cada clase
plt.figure(figsize=(20,20))


columns_object = ['Host Since', 'Host Response Time', 'Host Verifications',
       'Neighbourhood', 'Neighbourhood Cleansed',
       'Neighbourhood Group Cleansed', 'Property Type', 'Room Type',
       'Bed Type', 'Amenities', 'Cancellation Policy', 'Features']


for i,feature in enumerate(airbnb_train.columns.drop(columns_object)):
    plt.subplot(6,5,i+1)   
    airbnb_train[feature].plot.hist(alpha=0.5, bins=100, grid = True)
    plt.legend()
    plt.title(feature)

plt.show()


# Para comprobar el histograma de una característica específica
#airbnb_train['Maximum Nights'].plot.hist(alpha=0.5, bins=100, grid = True)


# In[1313]:


# Comandos para analizar cada una de las caracteristicas. 
print((airbnb_train['Maximum Nights'] > 1200).value_counts())
numberT = (airbnb_train['Maximum Nights'] > 1200).values.sum()
print ((numberT / 10565)*100)
print(airbnb_train['Maximum Nights'].value_counts().sum())


# In[1314]:


# Aplicamos el plot de 'scatter' para analizar mejor los outliers en comparación a la característica target
# Comandos para analizar cada una de las características
plt.figure(figsize=(1000,1000))
airbnb_train.plot(kind = 'scatter',x='Maximum Nights',y = 'Price')
plt.xlabel('Maximum Nights ($m^2$)')
plt.ylabel('price (€)')
plt.show()


# Usando la función ```describe()``` y realizando el histograma de las 30 características numéricas, vemos que puede haber 'outliers' para las variables:
# - *Host Listing Counts* , *Host Total Listings Counts*. Además hay una tercera variable *Calculated host listings count* que parece decir la misma información, pero esperaremos a ver su correlación para tomar alguna decisión más fiable. 
# - *Accommodates* , *Price*, *Security Deposit*, *Cleaning Fee*, *Guests Included*, *Extra People*, *Minimum Nights*, *Maximum Nights*, *Number of reviews*, y tal vez *Reviews per Month*
# 
# Procedemos a analizar cada una de estas variables y tomando decisiones para cada una de ellas. Encontramos un valor de corte que no supere el 10% de los valores totales de la columna. 
# 
# - *Host Listing Counts* para un filtro mayor de 100 eliminio 288 muestras (2.72%)
# - *Host Total Listing Counts* para un filtro mayor de de 100 eliminio 288 muestras (2.72%)
# - *Calculated host listing count* para un filtro mayor de 60 elimino 317 muestras (3.00%)
# - ~~*Accomodates*~~ para el valor de 16 : 27 registros ; 15 : 1 registro ; 14 : 6 registros. No voy a aplicar ningún filtro para esta variable, ya que puede haber alojamientos que acepten 16 personas. 
# - *Price* analizando el histograma, veo que hay registros claramente aislados a partir de 600€, pero viendo los valores de la tabla de describe, siendo la media de precio 50€, y sabiendo que el 75% de los valores marcan los 80€, he probado varios filtros. Si pongo el valor de corte en 300€ que todavía es un valor razonable, estaría eliminando 84 registros que representan 0.79% de todos los registros. Por lo que realizando el corte en 300€ probablemente tengamos mejores predicciones (la representación de datos es más compacta) a costa de que menos de 1 registro en cada 100 no se van a poder estimar. 
# - *Security Deposit* , entiendo que los valores de deposito son mas altos que los precios del alquiler. Aunque a veces son desorbitados, si pongo el corte en 400€, eliminaría 195 registros (1.84%) y si pongo el corte en 500€, eliminaría 55 registros (0.52%) . La media está en 150€ y el tercer quartil marca 200€
# - *Cleaning Fee*, podriamos poner el corte en 100€, siendo la media 25€, esto eliminaría 79 registros, que corresponde al 0.74%
# - ~~*Guests Included*~~, al igual que con la variable Accommodates, no considero muy relevante aplicar filtros para esta variable. Los outliers podrian estar en 16 invitados con 1 registro, igual para 14 invitados, o 2 registros para 12 y 15 invitados. 
# - *Extra People* , en este caso el gráico de scatter, da una mejor una visión de los outliers, realizando el corte en 45 (ya me parecen demasiados invitados extra!) elimino 52 registros (0.49% del dataset)- El tercer quartil nos indica 14 personas. Realmente tiene sentido bajar este número, pero si ponemos el corte en 20 personas, estaria eliminando 612 registros (5.79%).
# - ~~*Minimun Nights*~~ , tal vez podríamos aplicar un corte a 50 dias, que supone 23 registros (0.22%).
# - *Maximum Nights*, seleccionando el corte por encima del tercer quartil (1200), supone 49 registros (0.46%). Al ser apartamentos para estancias temporales no tiene sentido que se puedan alquilar por mucho mas de 1200 dias - 4 año.  
# - ~~*Number of Reviews*~~ , analizando los datos no considero importante aplicar un filtro para esta variable, ya que realmente puede ser que haya pocos alojamientos que tengan muchas reviews. 
# - ~~*Reviews per Month*~~ , analizando los datos no considero relevante aplicar filtros para esta variable tampoco.
# 
# Para evaluar cuántos filtros aplico, voy a evaluar cuáles son las variables más importantes y después tomar decisiones.    
# (En una segunda fase he probado a aplicar el filtro 6 , y con Lasso, obtengo un coefficiente de 0.002 para la característica *Maximum Nights* lo que me indica que puedo eliminar esta característica. El accuracy del modelo baja muy poquito, pero me tiene más sentido trabajar con valores reales (Maximo numero de noches 1200), ya que veo dificil alquilar un apartamento vacaciones/temporal trabajo por mas de 4 años. Además, con este filtro solo elimino 49 registros, que es un número pequeño de registros a eliminar. Lo elimino después de la matriz de correlación)

# In[1315]:


# Aplico filtros
f1 = airbnb_train['Calculated host listings count'] < 60   # 3.00%
f2 = airbnb_train['Price'] < 300                           # 0.79%
f3 = airbnb_train['Security Deposit'] < 400                # 1.84%
f4 = airbnb_train['Cleaning Fee'] < 100                    # 0.74%
f5 = airbnb_train['Extra People'] < 30                     # 5.79%
f6 = airbnb_train['Maximum Nights'] < 1200                 # 0.46%

#airbnb_train = airbnb_train[f1]
airbnb_train = airbnb_train[f2]
#airbnb_train = airbnb_train[f3]
#airbnb_train = airbnb_train[f4]
#airbnb_train = airbnb_train[f5]
airbnb_train = airbnb_train[f6]


# In[1316]:


airbnb_train.shape


# Si aplicamos todos los filtros nos quedamos con 953 registros de 10565.
# Decido de momento aplicar solo el filtro del precio, y después si hay alguna característica que sea importante en comparación con la target, puedo revisar y aplicar algún filtro adicional.
# De esta forma solo se han eliminado 111 registros 
# 
# Una opción era considerar el precio total, sumando *Cleaning Fee*, pero viendo que hay 40% NAN, si aplico el filtro antes de imputar los NAN, se eliminan muchos registros, por lo que voy a esperar a analizar la importancia de esta caracteristica para tomar una decisión.  
# 
# Vamos a proceder a reemplazar los NA valores para cada característica por la media de los valores en esa característica.

# In[1317]:


# Buscamos todas las características numéricas y contamos el numero de NAs
numeric_feature_mask = airbnb_train.dtypes== float
numeric_cols = airbnb_train.columns[numeric_feature_mask].tolist()
print(airbnb_train[numeric_cols].isna().sum(axis=0))


# In[1318]:


# Para las columnas de Cleaning Fee y Security Deposit, vamos a asumir que si no tienen valor, es porque el precio es 0€
airbnb_train['Cleaning Fee'].fillna(0, inplace=True) 
airbnb_train['Security Deposit'].fillna(0, inplace=True)


# In[1319]:


# Ahora procedemos a reemplazar en el resto de variables numéricas, los NA por el valor media 
# para cada una de las características
for c in numeric_cols:
    if airbnb_train[c].isna != 0:
        airbnb_train[c].fillna(airbnb_train[c].mean(), inplace=True)

#print(airbnb_train[numeric_cols].isna().sum(axis=0))


# #### 2.5 Análisis Características 'object'

# In[1320]:


# Ver las características que son categóricas 
categorical_feature_mask = airbnb_train.dtypes==object
categorical_cols = airbnb_train.columns[categorical_feature_mask].tolist()
print(categorical_cols)

#print(airbnb_train['Property Type'].value_counts())
#print(airbnb_train['Property Type'].isna().sum(axis=0))
airbnb_train[categorical_cols].isna().sum(axis=0)


# Convertimos *Host Since* a los años que el 'host' (dueño) lleva como miembro en airbnb

# In[1321]:


from datetime import datetime

#Primero voy a reemplazar los 2 x NA por la moda
airbnb_train["Host Since"].fillna(airbnb_train["Host Since"].mode()[0], inplace=True)

# Transformar esta columna a formato fecha y poder extraer el año para calcular los años de permanencia como miembro
airbnb_train['Host Since'] = airbnb_train['Host Since'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d'))
airbnb_train['Host Years Membership'] = airbnb_train['Host Since'].apply(lambda x: 2020 - x.year)

# Pasar la ultima columna a la primera
cols =  airbnb_train.columns.tolist()
cols = cols[-1:] + cols[:-1]
airbnb_train = airbnb_train[cols]

# Eliminar la columna Host Since
airbnb_train.drop('Host Since' , inplace=True, axis=1 )
airbnb_train.head()


# ##### 2.5.1 - Generación de nuevas características

# Para contabilizar estas variables en nuestro modelo, vamos a proceder a contar el numero de características que hay en cada columna de 'Host Verifications', 'Amenities' y 'Features' , de tal forma que cuanta mas tenga, es un indicativo de mayor número de prestaciones. 
# - 'Host verifications' tiene numeración de 1 a 10
# - 'Amenities' tiene numeración de 1 a 34
# - 'Features' tiene numeración de 1 a 8

# In[1322]:


# Vamos a rellenar los NA con vacios, suponiendo no hay muchos, y suponemos que si no han escrito, es porque no tienen 
# En Host verification hay 6 , y en Amenities hay 82
airbnb_train['Host Verifications'].fillna("", inplace=True)
airbnb_train['Amenities'].fillna("", inplace=True)
             
airbnb_train['Host Verifications'] = airbnb_train['Host Verifications'].apply(lambda x: len(str(x).split(',')))
airbnb_train['Amenities'] = airbnb_train['Amenities'].apply(lambda x: len(str(x).split(',')))
airbnb_train['Features'] = airbnb_train['Features'].apply(lambda x: len(str(x).split(',')))


# In[1323]:


airbnb_train.head()


# In[1324]:


# Caracteristicas: 'Neighbourhood', 'Neighbourhood Cleansed', 'Neighbourhood Group Cleansed'
# Voy a hacer un análisis de estas tres variables

df_aux = airbnb_train[['Neighbourhood', 'Neighbourhood Cleansed','Neighbourhood Group Cleansed']].copy() 
df_aux.isna().sum()


# In[1325]:


df_aux.head(100)


# In[1326]:


#df_aux['Neighbourhood Group Cleansed'].value_counts()
#print(df_aux['Neighbourhood'].unique())
#print(df_aux['Neighbourhood Cleansed'].unique())
#print(df_aux['Neighbourhood Group Cleansed'].unique())


# La columna de *Neighbourhood* tiene 66 barrios diferentes y 3526 valores vacíos. 
# La columna de *Neighbourhood Cleansed* tiene 125 barrios diferentes y 0 valores vacíos. 
# La columna de *Neighbourhood Group Cleansed* tiene 21 barrios diferentes y 0 valores vacíos. 
# 
# Analizando las tres columnas y viendo los códigos postales, *Neighbourhood Cleansed*,tiene demasiadas clasificaciones de barrios. 
# La variable de *Neigbourhood*, tiene bastantes registros sin rellenar. Una idea sería poder analizar los *Zipcode* y realizar una correspondencia para rellenar los vacíos en esta columna (pero esto lo dejo para una segunda fase, o mejora de la practica :)). 
# 
# Pero viendo los distintos barrios en *Neighbourhood Group Cleansed* me parece razonable usar esta columna, a que al final engloba barrios por regiones.
# 
# Por lo que voy a eliminar las otras dos columnas del data set. 

# In[1327]:


airbnb_train.drop(['Neighbourhood', 'Neighbourhood Cleansed'], inplace=True, axis=1)


# ##### 2.5.2 Codificación de características categóricas
# 
# Vamos a realizar una codificación de de las características categóricas basándonos en el método de MEAN ENCODER/ TARGET-BASED ENCODER, que calcula la probabilidad de cada categoria dentro de una columna, basándose en los valores de la columna target, en nuestro caso *Price* , y usa ese valor númerico para representar esa categoría.    
# 
# Antes de proceder con la codificación, vamos a revisar que todas los registros en nuestro dataset tienen valores. En caso negativo usaremos la moda de entre los valores de esa caracteristica para reemplazar estos valores vacíos. 

# In[1328]:


# Reemplazar los valores NA de cada caracteristica por la moda
categorical_feature_mask = airbnb_train.dtypes==object
categorical_cols = airbnb_train.columns[categorical_feature_mask].tolist()
airbnb_train[categorical_cols].isna().sum(axis=0)


# In[1329]:


airbnb_train["Host Response Time"].fillna(airbnb_train["Host Response Time"].mode()[0], inplace=True)


# In[1330]:


# Creamos un dictionario que mapea el la media en valor para cada una de las categorias dentro de cada caracteristica.
# Realizamos la categorizacion usandoo el métoodo del Mean Encoder / target Encoder
categorical = ['Host Response Time', 'Property Type', 'Room Type', 'Bed Type', 'Cancellation Policy', 'Neighbourhood Group Cleansed']
# Es lo mismo que usar la lista de categorical_cols

mean_map = {}
for c in categorical:
    mean = airbnb_train.groupby(c)['Price'].mean()
    airbnb_train[c] = airbnb_train[c].map(mean)    
    mean_map[c] = mean
    
#Usaremos el dictionary mean_map para despues aplicarlo al airbnb_test set.


# In[1331]:


pd.set_option('display.max_columns', None)
airbnb_train.head()


# ### 3. Análisis entre variables
# 
# Vamos a estudiar la correlacion entre las variables respecto a la varible 'target' que es 'Price'. Esto nos va ayudar a indentificar las caracteristicas que estén altamente correlacionadas, y por tanto podemos eliminar. 
# 
# Si la correlación entre dos características es muy grande, puede crear errores en algunos algoritmos de machine learning, como por ejemplo en el caso de la regresión lineal.
# 
# Para evitar esto, vamos a eliminar las características que muestrn un |𝜌|>0.8

# In[1332]:


first_col = airbnb_train.pop('Price')
airbnb_train.insert(0, 'Price', first_col)


# In[1333]:


airbnb_train.describe().T


# In[1334]:


#Matriz de correlación 
airbnb_train.corr()


# In[1335]:


import seaborn as sns

# Compute the correlation matrix
corr = airbnb_train.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(12, 10))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask,vmin = 0.0, vmax=1.0, center=0.5,
            linewidths=.1, cmap="YlGnBu", cbar_kws={"shrink": .8})

plt.show()


# - Vemos que *Availability 30, 60 y 90* estan altamente correlacionadas entre si, y decido quedarme con *Avalability 30*
# - *Host Listing Count* y *Host Total Listing Count* tienen una correlación muy alta con *Calculated host listing count* , por lo que voy a eliminar estas dos características 
# - *Beds* y *Accomodates* tiene una correlación de 0.8, pero *Accomodates* tiene mayor peso sobre *Price* por lo que elimino *Beds*
# - Review Scores Value esta altamente correlacionada con Review Scores Rating (0.79), pero voy a esperar el resultado de Lasso para ver si hay mas relaciones con el resto de características 'Review Scores'

# In[1336]:


airbnb_train.drop(['Availability 60' , 'Availability 90', 'Host Listings Count' ,
                   'Host Total Listings Count' , 'Beds'] , inplace= True , axis=1 )


# In[1337]:


len(airbnb_train.columns)


# In[1338]:


import seaborn as sns

# Compute the correlation matrix
corr = airbnb_train.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(12, 10))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask,vmin = 0.0, vmax=1.0, center=0.5,
            linewidths=.1, cmap="YlGnBu", cbar_kws={"shrink": .8})

plt.show()


# Después de haber analizado la dimensionalidad del data set con Lasso y RandomForest, procedemos a realizar algunos cambios: 
# 
# -	Para RandomForest , las variable BedType no tiene importancia y viendo el coeficiente de esta variable para Lasso  (0.4) , me parece bien hacer un corte en Lasso de eliminar las características con coeficiente < 0.5, siendo la siguiente Lista : *Host Response Time* , *Property type* , *Minimum_Nights* , *Latitude*, *Bed Type*, *Availability 365*
# -	He visto que las características de Reviews Scores – Rating, Accuracy, Cleanliness, Checkin, Communication y Location,  estan de alguna forma correlacionas con Review Scores Value (desde 0.79 hasta 0.42 la menor) voy a probar a hacer una media de todas estas características y ver si se mejora el modelo. Voy a ver también el peso de estas características con nuestro target *Price*.   

# Los valores de correlación de las características de Review Scores con respecto a Price son los siguientes: 
# - Review Scores Rating - 0.056093
# - Review Scores Accuracy - 0.057848
# - Review Scores Cleanliness - 0.083447
# - Review Scores Checkin - -0.009747
# - Review Scores Communication - 0.007708
# - Review Scores Location - 0.141600
# - Review Scores Value- 0.023743
# 
# Voy a realizar la media de estos siete valores y crear una característica nueva que se llama *Review Scores Mean* para eliminar después esta 7 características.

# In[1339]:


airbnb_train['Review Scores Mean'] = airbnb_train[['Review Scores Rating', 'Review Scores Accuracy' ,
                                              'Review Scores Cleanliness', 'Review Scores Checkin',
                                              'Review Scores Communication', 'Review Scores Location',
                                              'Review Scores Value']].mean(axis=1)

airbnb_train.head()


# In[1340]:


# Fase 2 - Eliminar las columnas después del primer análisis con lasso y RamdonForest
airbnb_train.drop(['Host Response Time' , 'Property Type', 'Minimum Nights' , 'Maximum Nights', 
                   'Latitude' , 'Bed Type', 'Availability 365', 'Review Scores Rating', 'Review Scores Accuracy' ,
                   'Review Scores Cleanliness', 'Review Scores Checkin', 'Review Scores Communication',
                   'Review Scores Location', 'Review Scores Value'] , inplace= True , axis=1 )


# In[1341]:


# Compute the correlation matrix
corr = airbnb_train.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(12, 10))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask,vmin = 0.0, vmax=1.0, center=0.5,
            linewidths=.1, cmap="YlGnBu", cbar_kws={"shrink": .8})

plt.show()


# Voy a probar a crear una característica en el valor cuadrado del número de baños, ya que esta característica es la quinta de importancia según RamdonForest

# In[1342]:


# Fase 3 - Crear una nueva característica
#airbnb_train['Cleaning Fee']= airbnb_train['Cleaning Fee'].apply(lambda x: x**2)


# ### 4. Preprocesamiento, Reducción de Dimensionalidad y comprobación de modelos ML

# #### 4.1 Preprocesado de airbnb_train
# 
# Se organizan los datos en el dataset de airbnb_train para poder usarlos en el modelo de Lasso que vamos a usar para ver si podemos reducir la dimensionalidad del dataset

# In[1343]:


from sklearn import preprocessing

data = airbnb_train.values
y_train = data[:,0:1]   # y_train es la primera columna, que es PRICE - nuestro target
#y_train = np.log10(data[:,0:1])    # y_train es la primera columna, que es PRICE - nuestro target - Aqui Aplico el logaritmo
X_train = data[:,1:]      # x_train es el resto de caracteristicas que van a definir nuestro target
feature_names = airbnb_train.columns[1:]

# Escalamos (con los datos de X_train)
scaler = preprocessing.StandardScaler().fit(X_train)
XtrainScaled = scaler.transform(X_train)


# #### 4.2 Applicar Lasso para evaluar la dimensionalidad del dataset
# 
# Para trabajar con el modelo de Lasso, es necesario obtener el valor alpha para el que una regression lineal pueda representar mejor los datos. Para ello usamos GridSearchCV e indicamos el número de 'folders' que queremos usar, vamos a probar con 10 folders.

# In[1344]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

alpha_vector = np.logspace(-1.5,0,20)
param_grid = {'alpha': alpha_vector }
grid = GridSearchCV(Lasso(), scoring= 'neg_mean_squared_error', param_grid=param_grid, cv = 10)
grid.fit(XtrainScaled, y_train)
print("best mean cross-validation score: {:.3f}".format(grid.best_score_))
print("best parameters: {}".format(grid.best_params_))

#-1 porque es negado
scores = -1*np.array(grid.cv_results_['mean_test_score'])
plt.semilogx(alpha_vector,scores,'-o')
plt.xlabel('alpha',fontsize=16)
plt.ylabel('10-Fold MSE')
plt.show()


# Una vez conseguido el valor para alpha, vamos a proceder a aplicar todas las transformaciones en el dataset de test, para evaluar el MSE y la precisión de nuestro modelo.

# #### 4.3 Aplicar todas las transformaciones de airbnb_train en airbnb_test

# In[1345]:


# Cargamos los datos 
airbnb_test = pd.read_csv('./data/test.csv', sep=';', decimal='.')

airbnb_test.head()
airbnb_test.shape


# In[1346]:


airbnb_test.isna().sum()


# In[1347]:


# Transformación de las caracteristicas Review Scores xxx en airibnb_test
airbnb_test['Review Scores Mean'] = airbnb_test[['Review Scores Rating', 'Review Scores Accuracy' ,
                                              'Review Scores Cleanliness', 'Review Scores Checkin',
                                              'Review Scores Communication', 'Review Scores Location',
                                              'Review Scores Value']].mean(axis=1)

# Transformación Fase 3 - característica de Bathroom**2 
#airbnb_test['Cleaning Fee'] = airbnb_test['Cleaning Fee'].apply(lambda x: x**2)


# In[1348]:


# Eliminar las caracteristicas que hemos desestimado
to_drop_location = ['Street', 'City', 'State', 'Market', 'Smart Location' , 'Country Code', 'Country' 
                    , 'Zipcode' , 'Geolocation']
to_drop_many_NA = ['Host Acceptance Rate', 'Square Feet', 'Has Availability', 'Jurisdiction Names', 
                   'License', 'Weekly Price' , 'Monthly Price' ]
to_drop_noMeaning = ['Host Location', 'Host Neighbourhood', 'Calendar Updated', 'First Review' ,'Last Review']

airbnb_test.drop(to_drop_location, inplace=True, axis=1)
airbnb_test.drop(to_drop_many_NA, inplace=True, axis=1)
airbnb_test.drop(to_drop_noMeaning, inplace=True, axis=1)

airbnb_test.drop(['Neighbourhood', 'Neighbourhood Cleansed'], inplace=True, axis=1)
airbnb_test.drop(['Availability 60' , 'Availability 90', 'Host Listings Count' , 
                  'Host Total Listings Count' , 'Beds'] , inplace= True , axis=1 )


airbnb_test.drop(['Host Response Time' , 'Property Type', 'Minimum Nights' , 'Maximum Nights', 
                   'Latitude' , 'Bed Type', 'Availability 365', 'Review Scores Rating', 'Review Scores Accuracy' ,
                   'Review Scores Cleanliness', 'Review Scores Checkin', 'Review Scores Communication',
                   'Review Scores Location', 'Review Scores Value'] , inplace= True , axis=1 )


# In[1349]:


len(airbnb_test.columns)
airbnb_test.isna().sum()


# In[1350]:


# Aplicar Filtro
f2_test = airbnb_test['Price'] < 300 
airbnb_test = airbnb_test[f2_test]

#f6_test = airbnb_train['Maximum Nights'] < 1200 
#airbnb_train = airbnb_train[f6_test]


# In[1351]:


# Tratar NAs

airbnb_test['Cleaning Fee'].fillna(0, inplace=True) 
airbnb_test['Security Deposit'].fillna(0, inplace=True)

numeric_feature_mask_test = airbnb_test.dtypes== float
numeric_col_test = airbnb_test.columns[numeric_feature_mask_test].tolist()


# Completar el resto de variables numericas con registros vacíos con el valor medio de los datos de training ojo!
for c in numeric_col_test:
    if airbnb_test[c].isna != 0:
        airbnb_test[c].fillna(airbnb_train[c].mean(), inplace=True)


# In[1352]:


# Convertir Host Since a númerico especificando el número de años que ha sido miembro - Host Years Membership 

#Primero voy a reemplazar los 2 x NA por la moda (training ojo!)
airbnb_test['Host Since'].fillna(airbnb_test['Host Since'].mode()[0], inplace=True)

# Transformar esta columna a formato fecha y poder extraer el año para calcular los años de permanencia como miembro
airbnb_test['Host Since'] = airbnb_test['Host Since'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d'))
airbnb_test['Host Years Membership'] = airbnb_test['Host Since'].apply(lambda x: 2020 - x.year)

# Pasar la ultima columna a la primera
cols_test =  airbnb_test.columns.tolist()
cols_test = cols_test[-1:] + cols_test[:-1]
airbnb_test = airbnb_test[cols_test]

# Eliminar la columna Host Since
airbnb_test.drop('Host Since' , inplace=True, axis=1 )


# In[1353]:


airbnb_test['Host Verifications'].fillna("", inplace=True)
airbnb_test['Amenities'].fillna("", inplace=True)
             
airbnb_test['Host Verifications'] = airbnb_test['Host Verifications'].apply(lambda x: len(str(x).split(',')))
airbnb_test['Amenities'] = airbnb_test['Amenities'].apply(lambda x: len(str(x).split(',')))
airbnb_test['Features'] = airbnb_test['Features'].apply(lambda x: len(str(x).split(',')))


# In[1354]:


#Categorizamos con los datos de airbnb_train (training ojo!)
# fase 2, se han eliminado 'Host Response Time', 'Property Type',Bed Type'
categorical = ['Room Type', 'Cancellation Policy', 'Neighbourhood Group Cleansed']

for c in categorical:
    airbnb_test[c] = airbnb_test[c].map(mean_map[c])


# In[1355]:


# Relleanar los registros vacíos despues de categorizar, usando la moda de esta característica en airbnb_train
# En la fase 2, se ha eliminado esta característica
#airbnb_test['Host Response Time'].fillna(airbnb_train['Host Response Time'].mode()[0], inplace=True)


# In[1356]:


# Colocamos la característica 'Price' en primera posición
first_col_test = airbnb_test.pop('Price')
airbnb_test.insert(0, 'Price', first_col_test)


# In[1357]:


print(airbnb_test.shape)
airbnb_test.head()


# De las variables categóricas : 
# - *Amenities* - 30 categorías de las 34 en airbnb_train
# - *Features* - 8 categorías de las 8 en airbnb_train
# - *Host Verifications* - 10 categorías de las 10 en airbnb_train

# In[1358]:


# Procesamiento y escalado dataset airbnb_test
data_test = airbnb_test.values
y_test = data_test[:,0:1]     # y_test es la primera columna, que es PRICE - nuestro target
X_test = data_test[:,1:]      # x_test es el resto de datos del dataframe
feature_names_test = airbnb_test.columns[1:]

# IMPORTANTE : esta normalización/escalado se realiza con el scaler anterior, basada en los datos de training
XtestScaled = scaler.transform(X_test) 


# #### 4.4 Aplicar Lasso

# In[1359]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

alpha_vector = np.logspace(-1.6, -0.4,20)
param_grid = {'alpha': alpha_vector }
grid = GridSearchCV(Lasso(), scoring= 'neg_mean_squared_error', param_grid=param_grid, cv = 10)
grid.fit(XtrainScaled, y_train)
print("best mean cross-validation score: {:.3f}".format(grid.best_score_))
print("best parameters: {}".format(grid.best_params_))

#-1 porque es negado
scores = -1*np.array(grid.cv_results_['mean_test_score'])
plt.semilogx(alpha_vector,scores,'-o')
plt.xlabel('alpha',fontsize=16)
plt.ylabel('10-Fold MSE')
plt.show()


# In[1360]:


from sklearn.metrics import mean_squared_error

alpha_optimo = grid.best_params_['alpha']
lasso = Lasso(alpha = alpha_optimo).fit(XtrainScaled,y_train)

ytrainLasso = lasso.predict(XtrainScaled)
ytestLasso  = lasso.predict(XtestScaled)
mseTrainModelLasso = mean_squared_error(y_train,ytrainLasso)
mseTestModelLasso = mean_squared_error(y_test,ytestLasso)

print('MSE Modelo Lasso (train): %0.3g' % mseTrainModelLasso)
print('MSE Modelo Lasso (test) : %0.3g' % mseTestModelLasso)

print('RMSE Modelo Lasso (train): %0.3g' % np.sqrt(mseTrainModelLasso))
print('RMSE Modelo Lasso (test) : %0.3g' % np.sqrt(mseTestModelLasso))

w = lasso.coef_
for f,wi in zip(feature_names,w):
    print(f,wi)


# In[1361]:


print("ACC (Train): ",lasso.score(XtrainScaled,y_train))
print("ACC (Test): ",lasso.score(XtestScaled,y_test))


# Vimos durante las clases que si la característica target está escorada, e intentamos alguna forma de transformarla para tener una distribución gaussiana, nuestro modelo puede realizar una mejor predición ajustando los parámetro de alpha. 
# 
# Para ello primero vemos como se comporta la distribución de esta característica cuando aplicamos el logaritmo de base 10

# In[1362]:


plt.hist(airbnb_train['Price'], bins=30)
plt.show()

plt.hist(np.log10(airbnb_train['Price']), bins=30)
plt.show()


# Al aplicar el logaritmo a nuestra característica *Price*, vemos que la distribuición de los datos sigue una representación gaussiana.
# 
# Vamos a volver a aplicar Lasso y ver si obtenemos mejores resultados. 

# In[1363]:


from sklearn import preprocessing

data = airbnb_train.values
#y_train = data[:,0:1]   # y_train es la primera columna, que es PRICE - nuestro target
y_train = np.log10(data[:,0:1])    # y_train es la primera columna, que es PRICE - nuestro target - Aqui Aplico el logaritmo
X_train = data[:,1:]      # x_train es el resto de caracteristicas que van a definir nuestro target
feature_names = airbnb_train.columns[1:]

# Escalamos (con los datos de X_train)
scaler = preprocessing.StandardScaler().fit(X_train)
XtrainScaled = scaler.transform(X_train)


# In[1364]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

alpha_vector = np.logspace(-4,-1,30)
param_grid = {'alpha': alpha_vector }
grid = GridSearchCV(Lasso(), scoring= 'neg_mean_squared_error', param_grid=param_grid, cv = 10)
grid.fit(XtrainScaled, y_train)
print("best mean cross-validation score: {:.3f}".format(grid.best_score_))
print("best parameters: {}".format(grid.best_params_))

#-1 porque es negado
scores = -1*np.array(grid.cv_results_['mean_test_score'])
plt.semilogx(alpha_vector,scores,'-o')
plt.xlabel('alpha',fontsize=16)
plt.ylabel('10-Fold MSE')
plt.show()


# In[1365]:


# Procesamiento y escalado dataset airbnb_test
data_test = airbnb_test.values
y_test = np.log10(data_test[:,0:1])    # y_test es la primera columna, que es PRICE - nuestro target
X_test = data_test[:,1:]      # x_test es el resto de datos del dataframe
feature_names_test = airbnb_test.columns[1:]

# IMPORTANTE : esta normalización/escalado se realiza con el scaler anterior, basada en los datos de training
XtestScaled = scaler.transform(X_test) 


# In[1366]:


from sklearn.metrics import mean_squared_error

alpha_optimo = grid.best_params_['alpha']
lasso = Lasso(alpha = alpha_optimo).fit(XtrainScaled,y_train)

ytrainLasso = lasso.predict(XtrainScaled)
ytestLasso  = lasso.predict(XtestScaled)

mseTrainModelLasso_exp = mean_squared_error(10**y_train,10**ytrainLasso)
mseTestModelLasso_exp = mean_squared_error(10**y_test,10**ytestLasso)

print('MSE Modelo Lasso (train): %0.3g' % mseTrainModelLasso_exp)
print('MSE Modelo Lasso (test) : %0.3g' % mseTestModelLasso_exp)

print('RMSE Modelo Lasso (train): %0.3g' % np.sqrt(mseTrainModelLasso_exp))
print('RMSE Modelo Lasso (test) : %0.3g' % np.sqrt(mseTestModelLasso_exp))


w = lasso.coef_
for f,wi in zip(feature_names,w):
    print(f,wi)


# In[1367]:


print("ACC (Train): ",lasso.score(XtrainScaled,y_train))
print("ACC (Test): ",lasso.score(XtestScaled,y_test))


# #### 4.5 Random Forest 
# 
# Aplicamos Random Forest, porque es una forma natural de seleccionar variables, por lo que nos puede ayudar a ver si se puede reducir la dimensionalidad. Usamos Random Forest en vez de un arbol de decisión para evitar overfitting. 

# In[1368]:


y_train = data[:,0:1]   # y_train es la primera columna, que es PRICE - nuestro target
X_train = data[:,1:]
y_test = data_test[:,0:1]   # y_test es la primera columna, que es PRICE - nuestro target
X_test = data_test[:,1:]


# In[1370]:


from sklearn.ensemble import RandomForestRegressor

# grid search
maxDepth = range(13,24)
tuned_parameters = {'max_depth': maxDepth}

grid = GridSearchCV(RandomForestRegressor(random_state=0, n_estimators=200, max_features='sqrt'), param_grid=tuned_parameters,cv=10, verbose=2) 
grid.fit(X_train, y_train)  # En arboles no hay que escalar los datos

print("best mean cross-validation score: {:.3f}".format(grid.best_score_))
print("best parameters: {}".format(grid.best_params_))

scores = np.array(grid.cv_results_['mean_test_score'])
plt.plot(maxDepth,scores,'-o')
plt.xlabel('max_depth')
plt.ylabel('10-fold ACC')

plt.show()


# Analizando los valores de maxDepthOptimo, GridSearchCV nos da un valor de 21, pero analizando la curva en rangos mayores y menores, cuanto menos profundidad tenga nuestro árbol, menos posibilidad de overfitting.     
# Vemos que una profundidad de 14, hemos reducido la precision en training de 0.96 a 0.91 , pero la predicción en test solo ha bajado de 0.77 a 0.76, por lo que elegimos 14 como parámetro de profundidad.

# In[1371]:


#maxDepthOptimo = grid.best_params_['max_depth']
maxDepthOptimo = 14
bagModel = RandomForestRegressor(max_depth=maxDepthOptimo,n_estimators=200,max_features='sqrt').fit(X_train,y_train)

print("Train: ",bagModel.score(X_train,y_train))
print("Test: ",bagModel.score(X_test,y_test))

# Para depth 23
Train:  0.9625695879699188
Test:  0.7743446899778894
    
# Para depth 20
Train:  0.9576425874787526
Test:  0.7721088654060326
    
# Para depth 18 
Train:  0.9481207746765632
Test:  0.7713440430803122
    
# Para depth 14
Train:  0.9080705212423533
Test:  0.7648940667015682
    
# FASE 2 
# Para depth 21 
Train:  0.9605328728033057
Test:  0.7746008245944278
    
# Para depth 14
Train:  0.9094093457517154
Test:  0.7674963032421871


# In[1372]:


importances = bagModel.feature_importances_
importances = importances / np.max(importances)
features = airbnb_train.columns.drop(['Price'])

indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,10))
plt.barh(range(X_train.shape[1]),importances[indices])
plt.yticks(range(X_train.shape[1]),features[indices])
plt.show()


# In[1373]:


rf = RandomForestClassifier(max_depth=20, n_estimators=200, max_features='sqrt')
rf.fit(X_train, y_train)

ytrainRandomforest = rf.predict(X_train)
ytestRandomforest = rf.predict(X_test)

mseTrainModelRF = mean_squared_error(y_train,ytrainRandomforest)
mseTestModelRF = mean_squared_error(y_test,ytestRandomforest)

print('MSE Modelo RandomForest (train): %0.3g' % mseTrainModelLasso)
print('MSE Modelo RandomForest (test): %0.3g' % mseTestModelRF)

print('RMSE Modelo RandomForest (train): %0.3g' % np.sqrt(mseTrainModelLasso))
print('RMSE Modelo RandoomForest (test) : %0.3g' % np.sqrt(mseTestModelRF))


# #### 4.6 SVM 
# 
# Vamos a probar ahora con un algoritmo SVM, de kernel RBF(radial Basis Function), ya que lineal, hemos probado Lasso, y RBF suele dar mejor resultado que polinómico.

# In[1129]:


from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

# Paso 2:
vectorC = np.logspace(1, 3, 10)
vectorG = np.logspace(-4, 1, 8)

param_grid = {'C': vectorC, 'gamma':vectorG}
grid = GridSearchCV(SVR(kernel='rbf'), param_grid=param_grid, cv = 5, verbose=2)
grid.fit(XtrainScaled, y_train)

print("best mean cross-validation score: {:.3f}".format(grid.best_score_))
print("best parameters: {}".format(grid.best_params_))

# Mostramos prestaciones en CV
scores = grid.cv_results_['mean_test_score'].reshape(len(vectorC),len(vectorG))

plt.figure(figsize=(10,6))
plt.imshow(scores, interpolation='nearest', vmin= 0.6, vmax=0.9)
plt.xlabel('log(gamma)')
plt.ylabel('log(C)')
plt.colorbar()
plt.xticks(np.arange(len(vectorG)), np.log10(vectorG), rotation=90)
plt.yticks(np.arange(len(vectorC)), np.log10(vectorC))
plt.title('5-fold accuracy')
plt.show()
print("best mean cross-validation score: {:.3f}".format(grid.best_score_))
print("best parameters: {}".format(grid.best_params_))


# In[1374]:


#Copt = grid.best_params_['C']
#Gopt = grid.best_params_['gamma']
Copt = 599.4842503189409
Gopt = 0.013894954943731374

svmModel = SVR(kernel='rbf',gamma = Gopt, C = Copt).fit(XtrainScaled,y_train)
print('Acc (TRAIN): %0.2f'%svmModel.score(XtrainScaled,y_train))
print('Acc (TEST): %0.2f'%svmModel.score(XtestScaled,y_test))

ytestSVM = svmModel.predict(XtestScaled)

mseTestModelSVM = mean_squared_error(y_test,ytestSVM)
print('MSE Modelo SVM (test): %0.3g' % mseTestModelSVM)
print('RMSE Modelo SVM (test) : %0.3g' % np.sqrt(mseTestModelSVM))


# #### 4.8 Conclusiones
# 
# Después de hacer la limpieza de datos he procedido a aplicar Lasso. Los resultados no son muy buenos MSE en test 659, lo que implica que hay un error de 25,7€ , siendo la media 50€, el error es alto. 
# Viendo que la caracterítica target estaba escorada a la izquierda en un histograma, y realizando una transformación con el logaritmo de base 10, se consigue una distribución gausiana de los datos, entendiendo que el modelo pueda dar una mejor respuesta, y sí es verdad pero se mejora muy poco, del orden de 1-3%.     
# **MSE para airbnb_test**:
# - LASSO - 681  (26,1€)
# - LASSO (log10(Price)) - 655  (25,6€)
# - Random Forest - 587   (24,9€)
# - SVM RB - 465 (21,6€)
# 
# 
# También he ejecutado un algoritmo de RamdomForest que me permite visualizar la importancia de las caracteristicas. Usando estos dos algoritmos, he llegado a la reducción de 8 caracteríticas, que tenian sentido tanto en Lasso, como en RamdomForest como mirando la matriz de correlacion en relación con la columna Target - *Price*.
# En esta mmisma fase, también vi si tenía alguna influencia aplicar algún filtro de entre las variables que habia seleccionado qeu podian tener outliers y podian por tanto afectar al resultado. La gran parte de los filtros eliminaba bastantes registros, en torno a un 2%, pero como no se solapaban los registros, este valor ascendia a aproximadamente 9%, por lo que he decicido solo aplicar el filtro para Maximum Nights, ya que la característica de Minimum Nights tenia un peso de 0 en Lasso, y temía que debido a los 'outliers', Maximun Nights estuviese teniendo un peso 'falso'. Al realizar este filtro, esta variable paso a tener un coefficiente muy bajo de 0.02, por lo que fue suficiente para eliminarla.   
# **MSE para airbnb_test**:
# - LASSO - 688  (26,2€)
# - LASSO (log10(Price)) - 678  (26€)
# - Random Forest - 614   (24,8€)
# - SVM RB - 452 (21,3€)
# 
# En la tercera fase, he probado a crear variables 'a la fuerza' para ver si mejoraban los resultados , especialmente el cuadrado de Bathrooms, Accommodates y Cleaning Fee. Me fijé en estas características, porque en la matriz de correlación tenian un peso fuerte, a parte de estar seleccionadas en las 6-top características con el algoritmo de RamdomForest. 
# Desafortunadamente, ninguna de ellas mejoró el modelo. 
# 
# 
# Por lo que con los datos presentes en el dataset, el mejor resultado ha sido con SVM RB, tras eliminar dimensionalidad y realizar transformaciones de variables como agrupar todas las características de Review. 
# 
# Mejoras:
# - Se puede intentar categorizar características como Ammenities/Features ponderando sus categorias, o realizando un estudio más exhaustivo de las opciones para crear otra variable que pueda describir mejor las facilidades que tiene el airbnb, de forma que se de mas peso a wifi, calefaccion central, TV.. etc
# - Para la variable creada de Reviews Scores Mean, se podría hacer una media ponderada
# 

# In[ ]:




