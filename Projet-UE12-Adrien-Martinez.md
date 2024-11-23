---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

```python
cities = pd.read_csv('cities.csv')
countries = pd.read_csv('countries.csv')
weather = pd.read_csv('daily-weather-cities.csv')
```

# Countries

```python
countries
```

```python
countries.set_index('country', inplace = True)
countries
```

```python
countries['Continent_code'] = countries['continent'].astype('category').cat.codes + 1
countries['pop_cat'] = pd.cut(countries['population'], bins = [0, 10**6, 5*10**6, 10*10**6, 25*10**6, 50*10**6, 100*10**6, 10**9], labels = [10, 15, 30, 50, 75, 100, 130])
countries
```

```python
countries.plot.scatter(x = 'capital_lng', y = 'capital_lat', s = 'pop_cat', c = 'Continent_code', cmap = 'viridis', xlabel = 'Longitude', ylabel = 'Latitude',
                       title = 'Représentation des capitales des 214 pays et territoires du monde,\n avec une taille proportionnelle à la population du pays');
```

L'attribution des couleurs a pour but de vérifier qu'à chaque pays a bien été attribué le continent qui lui correspond. On remarque qu'à certains territoires, tels que la Zambie, ont été attribués les mauvais continents.

```python
# Correction du continent de la Zambie
countries.loc['Zambia', 'continent'] = 'Africa'
countries.loc['Zambia', 'continent']
```

```python
countries.sort_values(by = 'area', ascending = True).plot(x = 'area', y = 'population', xlabel = 'Area', ylabel = 'Population', title = "Etude d'un lien de corrélation potentiel entre la population et la spuerficie d'un pays \n");
```

Ce graphique nous montre que la population d'un pays n'est pas directement liée à sa superficie. Nous allons à présent étudier le lien entre la densité d'un pays et sa superficie. 

```python
countries['Density'] = countries['population']/countries['area']
by_density = countries.sort_values(by = 'Density', ascending = False)
by_density.head(20)
```

```python
by_density.tail(30)
```

```python
countries['Density'].max()/countries['Density'].min()
```

Alors que les pays les plus denses ont très majoritairement une superficie très faible, les pays les moins denses n'ont pas nécessairement une superficie très importante : les Terres australes et antarctiques françaises ainsi que les Iles Falkland, l'Islande et le Suriname en font partie alors qu'ils ont une superficie relativement petite (moins de 200 000 km²). On a par ailleurs montré que le pays le plus densément peuplé (Macao) est environ 1 164 000 fois plus dense que le pays le moins densément peuplé (l'Antarctique française), un rapport assez impressionnant !


# Cities

```python
cities.set_index('station_id', inplace = True)
cities
```

# Weather
## Etude différenciée de l'évolution des températures à _Paris_, _Vienne_, _Bruxelles_ et _Berlin_
````{admonition} →
 L'objectif de cette étude est de mettre en évidence les conséquences des activités humaines au cours du dernier siècle (notamment les émissions de gaz à effet de serre qui y sont associés) sur le climat de ces 4 capitales d'Europe de l'Ouest, en s'intéressant aux variations de température dans ces villes.

```python
weather['date'] = pd.to_datetime(weather['date'], format = 'ISO8601')
weather['city_name'].unique()
```

```python
paris = weather[weather['city_name'] == 'Paris']
vienna = weather[weather['city_name'] == 'Vienna']
brussels = weather[weather['city_name'] == 'Brussels']
berlin = weather[weather['city_name'] == 'Berlin']
```

## Augmentation des températures moyennes

```python
paris['Year'] = paris['date'].dt.to_period('Y') # Utile pour plus tard
paris.set_index('date', inplace = True)

vienna['Year'] = vienna['date'].dt.to_period('Y')
vienna.set_index('date', inplace = True)

brussels['Year'] = brussels['date'].dt.to_period('Y')
brussels.set_index('date', inplace = True)

berlin['Year'] = berlin['date'].dt.to_period('Y')
berlin.set_index('date', inplace = True)
```

```python
paris_avg_temp = paris['avg_temp_c'].rolling(window = pd.Timedelta(260, 'W')).mean()
paris_avg_temp.plot(x = 'date', y = 'avg_temp_c', xlabel = 'Date', ylabel = 'Température moyenne sur 5 ans en °C', title = 'Evolution de la température moyenne sur 5 ans entre 1945 et 2023 à Paris \n');
plt.show()

vienna_avg_temp = vienna['avg_temp_c'].rolling(window = pd.Timedelta(260, 'W')).mean()
vienna_avg_temp.plot(x = 'date', y = 'avg_temp_c', xlabel = 'Date', ylabel = 'Température moyenne sur 5 ans en °C', title = 'Evolution de la température moyenne sur 5 ans entre 1950 et 2023 à Vienne \n');
plt.show()

brussels_avg_temp = brussels['avg_temp_c'].rolling(window = pd.Timedelta(260, 'W')).mean()
brussels_avg_temp.plot(x = 'date', y = 'avg_temp_c', xlabel = 'Date', ylabel = 'Température moyenne sur 5 ans en °C', title = 'Evolution de la température moyenne sur 5 ans entre 1970 et 2023 à Bruxelles \n');
plt.show()

berlin_avg_temp = berlin['avg_temp_c'].rolling(window = pd.Timedelta(260, 'W')).mean()
berlin_avg_temp.plot(x = 'date', y = 'avg_temp_c', xlabel = 'Date', ylabel = 'Température moyenne sur 5 ans en °C', title = 'Evolution de la température moyenne sur 5 ans entre 1930 et 2023 à Berlin \n');
plt.show()
```

On remarque une tendance commune à l'augmentation des températures moyennes annuelles, notamment à partir des 1980, ce qui coïncide avec une activité industrielle et agricole intense en Europe. Cette augmentation est particulièrement marquée à Berlin et à Vienne, c'est-à-dire dans les plus centrales de ces 4 capitales.

```python
paris.loc['2010':'2010']
```

```python
paris.loc['2010':'2010', 'avg_temp_c':'avg_temp_c'].count()
```

Par ailleurs, il semblerait que c'est à cause du manque de valeurs que l'on remarque des pics importants qui semblent incohérents dans les courbes (comme dans les années 2010 sur la courbe de Paris).

```python
paris.boxplot(['avg_temp_c'], by = 'Year');
print('Paris :')
plt.show()

vienna.boxplot(['avg_temp_c'], by = 'Year');
print('Vienne :')
plt.show()

brussels.boxplot(['avg_temp_c'], by = 'Year');
print('Bruxelles :')
plt.show()

berlin.boxplot(['avg_temp_c'], by = 'Year');
print('Berlin :')
plt.show()
```

On constate également sur cette représentation une augmentation des moyennes annuelles, avec une intensification des évènements de fortes chaleurs, notamment à Bruxelles, Vienne et Paris.


# Augmentation des températures extrêmes

```python
paris['pseudo ecart_type'] = np.sqrt(0.5*((paris['min_temp_c']-paris['avg_temp_c'])**2 + (paris['max_temp_c']-paris['avg_temp_c'])**2))
vienna['pseudo ecart_type'] = np.sqrt(0.5*((vienna['min_temp_c']-vienna['avg_temp_c'])**2 + (vienna['max_temp_c']-vienna['avg_temp_c'])**2))
brussels['pseudo ecart_type'] = np.sqrt(0.5*((brussels['min_temp_c']-brussels['avg_temp_c'])**2 + (brussels['max_temp_c']-brussels['avg_temp_c'])**2))
berlin['pseudo ecart_type'] = np.sqrt(0.5*((berlin['min_temp_c']-berlin['avg_temp_c'])**2 + (berlin['max_temp_c']-berlin['avg_temp_c'])**2))
```

```python
paris_min_temp = paris['avg_temp_c'].resample(rule = pd.Timedelta(52, 'W')).min()
paris_min_temp.plot(x = 'date', y = 'min_temp_c', xlabel = 'Date', ylabel = 'Température minimale sur 1 an en °C', title = 'Evolution de la température minimale sur 1 an entre 1945 et 2023 à Paris \n');
plt.show()

vienna_min_temp = vienna['avg_temp_c'].resample(rule = pd.Timedelta(52, 'W')).min()
vienna_min_temp.plot(x = 'date', y = 'min_temp_c', xlabel = 'Date', ylabel = 'Température minimale sur 1 an en °C', title = 'Evolution de la température minimale sur 1 an entre 1950 et 2023 à Vienne \n');
plt.show()

brussels_min_temp = brussels['avg_temp_c'].resample(rule = pd.Timedelta(52, 'W')).min()
brussels_min_temp.plot(x = 'date', y = 'min_temp_c', xlabel = 'Date', ylabel = 'Température minimale sur 1 an en °C', title = 'Evolution de la température minimale sur 1 an entre 1965 et 2023 à Bruxelles \n');
plt.show()

berlin_min_temp = berlin['avg_temp_c'].resample(rule = pd.Timedelta(52, 'W')).min()
berlin_min_temp.plot(x = 'date', y = 'min_temp_c', xlabel = 'Date', ylabel = 'Température minimale sur 1 an en °C', title = 'Evolution de la température minimale sur 1 an entre 1930 et 2023 à Berlin \n');
plt.show()
```

```python
paris_max_temp = paris['avg_temp_c'].resample(rule = pd.Timedelta(52, 'W')).max()
paris_max_temp.plot(x = 'date', y = 'max_temp_c', xlabel = 'Date', ylabel = 'Température maximale sur 1 an en °C', title = 'Evolution de la température maximale sur 1 an entre 1945 et 2023 à Paris \n');
plt.show()

vienna_max_temp = vienna['avg_temp_c'].resample(rule = pd.Timedelta(52, 'W')).max()
vienna_max_temp.plot(x = 'date', y = 'max_temp_c', xlabel = 'Date', ylabel = 'Température maximale sur 1 an en °C', title = 'Evolution de la température maximale sur 1 an entre 1950 et 2023 à Vienne \n');
plt.show()

brussels_max_temp = brussels['avg_temp_c'].resample(rule = pd.Timedelta(52, 'W')).max()
brussels_max_temp.plot(x = 'date', y = 'max_temp_c', xlabel = 'Date', ylabel = 'Température maximale sur 1 an en °C', title = 'Evolution de la température maximale sur 1 an entre 1950 et 2023 à Bruxelles \n');
plt.show()

berlin_max_temp = berlin['avg_temp_c'].resample(rule = pd.Timedelta(52, 'W')).max()
berlin_max_temp.plot(x = 'date', y = 'max_temp_c', xlabel = 'Date', ylabel = 'Température maximale sur 1 an en °C', title = 'Evolution de la température maximale sur 1 an entre 1930 et 2023 à Berlin \n');
plt.show()
```

On ne remarque pas d'augmentation particulière de la fréquence des évènements très froids, ils sont cependant plus intenses (comme par exemple à Paris), avec une tendance globale à l'augmentation de la température minimale (comme le montrent les graphes de Paris, Vienne et Bruxelles).
En revanche, on remarque une nette augmentation et intensification des évènements très chauds dans toutes les villes, notamment à Vienne et à Bruxelles.

```python
paris.groupby('Year')['avg_temp_c'].std().plot(title = "Evolution de l'écart-type sur un an des moyennes journalières des températures à Paris de 1944 à 2023");
plt.show()

vienna.groupby('Year')['avg_temp_c'].std().plot(title = "Evolution de l'écart-type sur un an des moyennes journalières des températures à Vienne de 1944 à 2023");
plt.show()

brussels.groupby('Year')['avg_temp_c'].std().plot(title = "Evolution de l'écart-type sur un an des moyennes journalières des températures à Bruxelles de 1944 à 2023");
plt.show()

berlin.groupby('Year')['avg_temp_c'].std().plot(title = "Evolution de l'écart-type sur un an des moyennes journalières des températures à Berlin de 1930 à 2023");
plt.show()
```

```python
paris.groupby('Year')['pseudo ecart_type'].mean().plot(title = "Evolution de la moyenne annuelle du pseudo-écart-type journalier des températures à Paris de 1944 à 2023");
plt.show()

vienna.groupby('Year')['pseudo ecart_type'].mean().plot(title = "Evolution de la moyenne annuelle du pseudo-écart-type journalier des températures à Vienne de 1944 à 2023");
plt.show()

brussels.groupby('Year')['pseudo ecart_type'].mean().plot(title = "Evolution de la moyenne annuelle du pseudo-écart-type journalier des températures à Bruxelles de 1970 à 2023");
plt.show()

berlin.groupby('Year')['pseudo ecart_type'].mean().plot(title = "Evolution de la moyenne annuelle du pseudo-écart-type journalier des températures à Berlin de 1930 à 2023");
plt.show()
```

Pour caractériser la dispersion des températures autour de la moyenne, on s'intéresse à deux indicateurs. Le premier est l'écart-type des températures moyennes quotidiennes dans une année, il ne permet pas de décrire correctement la dispersion des températures autour des valeurs moyennes quotidiennes car elle est basée sur des séries de valeurs moyennes. Le second est une valeur artificiellement créée, appelé 'pseudo-écart-type', il permet de caractériser la dispersion des températures maximales et minimales quotidiennes autour des températures moyennes correspondant au même jour. On constate que cette dispersion a une tendance globale à l'augmentation dans ces quatre villes, ce qui permet bien de démontrer le progrès du réchauffement climatique.
