import requests, pandas as pd
from bs4 import BeautifulSoup

url = 'https://www.capfriendly.com/ajax/signings/'

teams = ['ducks', 'coyotes', 'bruins', 'sabres', 'flames', 'hurricanes', 'blackhawks', 'avalanche', 'bluejackets', 'stars', 'redwings', 'oilers', 'panthers', 'kings', 'wild', 'canadiens', 'predators', 'devils', 'islanders', 'rangers', 'senators', 'flyers', 'penguins', 'sharks', 'blues', 'lightning', 'mapleleafs', 'canucks', 'goldenknights', 'capitals', 'jets']

def get_team_contracts(team):
    data = []
    for i in range(1,10):
        try:
            r = requests.get(url + team + '?page=' + str(i))
            s = r.json()['data']['html']
            soup = BeautifulSoup(s, 'html.parser')
            table_rows = soup.find_all('tr')
            columns = ['Name', 'Age', 'Position', 'Team', 'Contract Start Date', 'Contract Type', 'NHL Clause', 'Way Clause', 'Term', 'Total Value', 'AAV']
            for row in table_rows:
                row_data = []
                for cell in row.find_all('td'):
                    text = cell.text.strip()
                    row_data.append(text)
                if len(row_data) == len(columns):
                    data.append(row_data)
        except:
            break
    df = pd.DataFrame(data, columns=columns)
    return df

def get_all_contracts():
    df = pd.DataFrame()
    for team in teams:
        print(team)
        df = df.append(get_team_contracts(team))
    return df

df = get_all_contracts()
df.to_csv('contracts.csv', index=False)