import pandas as pd
import numpy as np


# Date is season end date e.x. 2025/2026 = 2026

df_2026 = pd.read_csv("E0.csv")
df_2026 = df_2026[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HS', 'AS']]

df_2025 = pd.read_csv("E1.csv")
df_2025 = df_2025[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HS', 'AS']]

df_2024 = pd.read_csv("E2.csv")
df_2024 = df_2024[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HS', 'AS']]

df_2023 = pd.read_csv("E3.csv")
df_2023 = df_2023[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HS', 'AS']]

df_2022 = pd.read_csv("E4.csv")
df_2022 = df_2022[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HS', 'AS']]


df_2022['Date'] = pd.to_datetime(df_2022['Date'], dayfirst=True)
df_2023['Date'] = pd.to_datetime(df_2023['Date'], dayfirst=True)
df_2024['Date'] = pd.to_datetime(df_2024['Date'], dayfirst=True)
df_2025['Date'] = pd.to_datetime(df_2025['Date'], dayfirst=True)
df_2026['Date'] = pd.to_datetime(df_2026['Date'], dayfirst=True)

df_all = pd.concat([df_2022, df_2023, df_2024, df_2025, df_2026]).sort_values('Date').reset_index(drop=True)


home_cols = {
    'HomeTeam': 'Team',
    'AwayTeam': 'Opponent',
    'FTHG': 'GoalsFor',
    'FTAG': 'GoalsAgainst',
    'HS': 'ShotsFor',
    'AS': 'ShotsAgainst'
}

away_cols = {
    'AwayTeam': 'Team',
    'HomeTeam': 'Opponent',
    'FTAG': 'GoalsFor',
    'FTHG': 'GoalsAgainst',
    'AS': 'ShotsFor',
    'HS': 'ShotsAgainst'
}



keep = ['Date', 'Team', 'Opponent', 'GoalsFor', 'GoalsAgainst', 'ShotsFor', 'ShotsAgainst', 'Venue', 'Result']

Home = df_all[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HS', 'AS', 'FTR']].rename(columns=home_cols)
Home['Venue'] = 'Home'

Away = df_all[['Date', 'HomeTeam', 'AwayTeam', 'FTAG', 'FTHG', 'AS', 'HS', 'FTR']].rename(columns=away_cols)
Away['Venue'] = 'Away'

result_map_home = {'H': 'W', 'D': 'D', 'A': 'L'}
result_map_away = {'A': 'W', 'D': 'D', 'H': 'L'}

Home['Result'] = Home['FTR'].map(result_map_home)
Away['Result'] = Away['FTR'].map(result_map_away)

Home = Home.drop(columns=['FTR'])
Away = Away.drop(columns=['FTR'])

team_df = pd.concat([Home, Away]).sort_values('Date').reset_index(drop=True)

team_df['Date'] = pd.to_datetime(team_df['Date'], dayfirst=True)
df_all['Date'] = pd.to_datetime(df_all['Date'], dayfirst=True)
team_df = team_df.sort_values('Date').reset_index(drop=True)


cols_to_roll = ['GoalsFor', 'GoalsAgainst', 'ShotsFor', 'ShotsAgainst']
rolling_col_names = ['RollingGoalsFor', 'RollingGoalsAgainst', 'RollingShotsFor', 'RollingShotsAgainst']

shifted = team_df.groupby('Team')[cols_to_roll].shift(1)
team_df[rolling_col_names] = shifted.groupby(team_df['Team']).rolling(10).mean().reset_index(level=0, drop=True)


home_stats = team_df[team_df['Venue'] == 'Home'][['Date', 'Team'] + rolling_col_names]
df_model = df_all.merge(home_stats, left_on=['Date', 'HomeTeam'], right_on=['Date', 'Team']).drop(columns='Team')

away_stats = team_df[team_df['Venue'] == 'Away'][['Date', 'Team'] + rolling_col_names]
df_model = df_model.merge(away_stats, left_on=['Date', 'AwayTeam'], right_on=['Date', 'Team'], suffixes=('_Home', '_Away')).drop(columns='Team')

# Testing model with the columns combined since difference in goals/shots is what matters
df_model['RollingGoalDif_Home'] = df_model['RollingGoalsFor_Home'] - df_model['RollingGoalsAgainst_Home']
df_model['RollingGoalDif_Away'] = df_model['RollingGoalsFor_Away'] - df_model['RollingGoalsAgainst_Away']
df_model['RollingShotDif_Home'] = df_model['RollingShotsFor_Home'] - df_model['RollingShotsAgainst_Home']
df_model['RollingShotDif_Away'] = df_model['RollingShotsFor_Away'] - df_model['RollingShotsAgainst_Away']

df_model.drop(columns=['RollingGoalsFor_Home', 'RollingGoalsAgainst_Home', 'RollingGoalsFor_Away', 'RollingGoalsAgainst_Away'], inplace=True)
df_model.drop(columns=['RollingShotsFor_Home', 'RollingShotsAgainst_Home', 'RollingShotsFor_Away', 'RollingShotsAgainst_Away'], inplace=True)

# save data
df_model.to_pickle("df_model.pkl") 



