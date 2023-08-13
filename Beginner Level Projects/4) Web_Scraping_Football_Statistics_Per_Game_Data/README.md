# Web Scraping Football Statistics Per Game Data

### About the Project
In this notebook I will describe the process of scraping data from web portal understat.com that has a lot of statistical information about all games in top 5 European football leagues.

From understat.com home page:

Expected goals (xG) is the new revolutionary football metric, which allows you to evaluate team and player performance.In a low-scoring game such as football, final match score does not provide a clear picture of performance.This is why more and more sports analytics turn to the advanced models like xG, which is a statistical measure of the quality of chances created and conceded.Our goal was to create the most precise method for shot quality evaluation.

For this case, we trained neural network prediction algorithms with the large dataset (>100,000 shots, over 10 parameters for each).On this site, you will find our detailed xG statistics for the top European leagues.

At this moment they have not only xG metric, but much more, that makes this site perfect for scraping statistical data about football games.

### import
```Python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import requests
from bs4 import BeautifulSoup
```

# Create urls for all seasons of all leagues

```Python
base_url = 'https://understat.com/league'
leagues = ['La_liga', 'EPL', 'Bundesliga', 'Serie_A', 'Ligue_1', 'RFPL']
seasons = ['2014', '2015', '2016', '2017', '2018', '2019', '2020','2021','2022']
```
# Working with JSON

```Python
import json

string_with_json_obj = ''

# Find data for teams
for el in scripts:
    if 'teamsData' in el.text:
      string_with_json_obj = el.text.strip()
      
# print(string_with_json_obj)

# strip unnecessary symbols and get only JSON data
ind_start = string_with_json_obj.index("('")+2
ind_end = string_with_json_obj.index("')")
json_data = string_with_json_obj[ind_start:ind_end]

json_data = json_data.encode('utf8').decode('unicode_escape')
```

# Understanding data with Python

```Python
data = json.loads(json_data)
print(data.keys())
print('='*50)
print(data['138'].keys())
print('='*50)
print(data['138']['id'])
print('='*50)
print(data['138']['title'])
print('='*50)
print(data['138']['history'][0])
```

![1](https://user-images.githubusercontent.com/77057546/185750685-61b75f0a-b729-4d67-85db-4bdf1f3ac4f6.png)

# Getting data for all teams

```Python

dataframes = {}
for id, team in teams.items():
  teams_data = []
  for row in data[id]['history']:
    teams_data.append(list(row.values()))
    
  df = pd.DataFrame(teams_data, columns=columns)
  dataframes[team] = df
  print('Added data for {}.'.format(team))
```

![2](https://user-images.githubusercontent.com/77057546/185750831-b90cc71b-54a7-4950-8ff2-1ee0adf294c1.png)

# Manipulations to make data as in the original source

```Python
for team, df in dataframes.items():
    dataframes[team]['ppda_coef'] = dataframes[team]['ppda'].apply(lambda x: x['att']/x['def'] if x['def'] != 0 else 0)
    dataframes[team]['ppda_att'] = dataframes[team]['ppda'].apply(lambda x: x['att'])
    dataframes[team]['ppda_def'] = dataframes[team]['ppda'].apply(lambda x: x['def'])
    dataframes[team]['oppda_coef'] = dataframes[team]['ppda_allowed'].apply(lambda x: x['att']/x['def'] if x['def'] != 0 else 0)
    dataframes[team]['oppda_att'] = dataframes[team]['ppda_allowed'].apply(lambda x: x['att'])
    dataframes[team]['oppda_def'] = dataframes[team]['ppda_allowed'].apply(lambda x: x['def'])
    
# And check how our new dataframes look based on Sevilla dataframe
dataframes['Sevilla'].head(2)
```

![3](https://user-images.githubusercontent.com/77057546/185750908-867c4333-160b-4aa6-a558-a41805502fb8.png)



```Python
frames = []
for team, df in dataframes.items():
    df['team'] = team
    frames.append(df)
    
full_stat = pd.concat(frames)
full_stat = full_stat.drop(['ppda', 'ppda_allowed'], axis=1)
full_stat.head(10)
```

![4](https://user-images.githubusercontent.com/77057546/185750969-7209ece2-950f-4bcc-99bb-3e9f0ac8ebcf.png)

# Scraping data for all teams of all leagues of all seasons


```Python
season_data = dict()
season_data[seasons[4]] = full_stat
print(season_data)
full_data = dict()
full_data[leagues[0]] = season_data
print(full_data)
```

```Python
full_data = dict()
for league in leagues:
  
  season_data = dict()
  for season in seasons:    
    url = base_url+'/'+league+'/'+season
    res = requests.get(url)
    soup = BeautifulSoup(res.content, "lxml")

    # Based on the structure of the webpage, I found that data is in the JSON variable, under <script> tags
    scripts = soup.find_all('script')
    
    string_with_json_obj = ''

    # Find data for teams
    for el in scripts:
        if 'teamsData' in el.text:
          string_with_json_obj = el.text.strip()

    # print(string_with_json_obj)

    # strip unnecessary symbols and get only JSON data
    ind_start = string_with_json_obj.index("('")+2
    ind_end = string_with_json_obj.index("')")
    json_data = string_with_json_obj[ind_start:ind_end]
    json_data = json_data.encode('utf8').decode('unicode_escape')
    
    
    # convert JSON data into Python dictionary
    data = json.loads(json_data)
    
    # Get teams and their relevant ids and put them into separate dictionary
    teams = {}
    for id in data.keys():
      teams[id] = data[id]['title']
      
    # EDA to get a feeling of how the JSON is structured
    # Column names are all the same, so we just use first element
    columns = []
    # Check the sample of values per each column
    values = []
    for id in data.keys():
      columns = list(data[id]['history'][0].keys())
      values = list(data[id]['history'][0].values())
      break
      
    # Getting data for all teams
    dataframes = {}
    for id, team in teams.items():
      teams_data = []
      for row in data[id]['history']:
        teams_data.append(list(row.values()))

      df = pd.DataFrame(teams_data, columns=columns)
      dataframes[team] = df
      # print('Added data for {}.'.format(team))
      
    
    for team, df in dataframes.items():
        dataframes[team]['ppda_coef'] = dataframes[team]['ppda'].apply(lambda x: x['att']/x['def'] if x['def'] != 0 else 0)
        dataframes[team]['ppda_att'] = dataframes[team]['ppda'].apply(lambda x: x['att'])
        dataframes[team]['ppda_def'] = dataframes[team]['ppda'].apply(lambda x: x['def'])
        dataframes[team]['oppda_coef'] = dataframes[team]['ppda_allowed'].apply(lambda x: x['att']/x['def'] if x['def'] != 0 else 0)
        dataframes[team]['oppda_att'] = dataframes[team]['ppda_allowed'].apply(lambda x: x['att'])
        dataframes[team]['oppda_def'] = dataframes[team]['ppda_allowed'].apply(lambda x: x['def'])
    
    frames = []
    for team, df in dataframes.items():
        df['team'] = team
        frames.append(df)
    
    full_stat = pd.concat(frames)
    full_stat = full_stat.drop(['ppda', 'ppda_allowed'], axis=1)
    
    full_stat['xG_diff'] = full_stat['xG'] - full_stat['scored']
    full_stat['xGA_diff'] = full_stat['xGA'] - full_stat['missed']
    full_stat['xpts_diff'] = full_stat['xpts'] - full_stat['pts']
    
    full_stat.reset_index(inplace=True, drop=True)
    season_data[season] = full_stat
  
  df_season = pd.concat(season_data)
  full_data[league] = df_season
  
data = pd.concat(full_data)
data.head()
  
```

![5](https://user-images.githubusercontent.com/77057546/185751094-59933b12-35d2-405a-89b8-16e57d660002.png)


```Python
data.index = data.index.droplevel(2)
data.index = data.index.rename(names=['league','year'], level=[0,1])
data.head()
```

![6](https://user-images.githubusercontent.com/77057546/185751232-bd0b7604-a240-4b28-ae0a-15795d6c7d5d.png)

```Python
data.tail()
```

![7](https://user-images.githubusercontent.com/77057546/185751268-598b8536-14b0-43d6-97d0-b258763ea218.png)

# Exporting data to CSV file

```Python
data.to_csv('Web_Scraping_Football_Statistics_understat_per_game.csv')
```

See you on another project.

## MAY THE POWER BE WITH YOU!!!

