import pandas as pd
import numpy as np
from datetime import datetime

fitbit_dataset1 = pd.read_json("./data/fitbit_sleep_data_1.json")
fitbit_dataset2 = pd.read_json("data/fitbit_sleep_data_2.json")
fitbit_dataset3 = pd.read_json("data/fitbit_sleep_data_3.json")
fitbit_dataset = pd.json_normalize(
    pd.concat([fitbit_dataset1, fitbit_dataset2, fitbit_dataset3], ignore_index=True).to_dict("records")
)

selected_columns = [
    "sleep.dateOfSleep",
    "sleep.minutesAsleep",
    "sleep.minutesAwake",
    "sleep.levels.data",
    "sleep.levels.summary.deep.minutes",
    "sleep.levels.summary.light.minutes",
    "sleep.levels.summary.rem.minutes"
]

selected_data = fitbit_dataset[selected_columns].copy()

flattened_data = []
for index, row in selected_data.iterrows():
    for item in row["sleep.levels.data"]:
        date_time = datetime.strptime(item["dateTime"], "%Y-%m-%dT%H:%M:%S.%f")
        flattened_row = {
            "date_of_sleep": row["sleep.dateOfSleep"],
            "asleep_seconds": row["sleep.minutesAsleep"]*60,
            "awake_seconds": row["sleep.minutesAwake"]*60,
            "category": item["level"],
            "time_hours": date_time.hour,
            "time_minutes": date_time.minute,
            "time_seconds": date_time.second,
            "seconds_count": item["seconds"],
            "deep_seconds": row["sleep.levels.summary.deep.minutes"] * 60,
            "light_seconds": row["sleep.levels.summary.light.minutes"] * 60,
            "rem_seconds": row["sleep.levels.summary.rem.minutes"]*60
        }
        flattened_data.append(flattened_row)

fitbit_df = pd.DataFrame(flattened_data)
fitbit_df.dropna(inplace=True)
print("Fitbit data is cleaned. Shape: ", fitbit_df.shape)

applewatch_df = pd.read_csv("./data/apple_watch_sleep_data.csv")
applewatch_df["Start Time"] = pd.to_datetime(applewatch_df["Start Time"])
applewatch_df["Date of Sleep"] = applewatch_df["Start Time"].dt.date

seconds_asleep = applewatch_df.loc[applewatch_df['Category'].isin(['Light', 'Deep', 'REM'])].groupby('Date of Sleep')['seconds'].sum().reset_index()
seconds_awake = applewatch_df.loc[applewatch_df['Category'] == 'Awake'].groupby('Date of Sleep')['seconds'].sum().reset_index()
light_seconds = applewatch_df.loc[applewatch_df['Category'] == 'Light'].groupby('Date of Sleep')['seconds'].sum().reset_index()
deep_seconds = applewatch_df.loc[applewatch_df['Category'] == 'Deep'].groupby('Date of Sleep')['seconds'].sum().reset_index()
rem_seconds = applewatch_df.loc[applewatch_df['Category'] == 'REM'].groupby('Date of Sleep')['seconds'].sum().reset_index()

applewatch_df = applewatch_df[applewatch_df['Category'] != 'Unspecified']

applewatch_df = pd.merge(applewatch_df, seconds_asleep, on='Date of Sleep', how='left', suffixes=('_total', '_asleep'))
applewatch_df.rename(columns={"seconds_total":"seconds"}, inplace=True)
applewatch_df = pd.merge(applewatch_df, seconds_awake, on='Date of Sleep', how='left', suffixes=('_total', '_wake'))
applewatch_df.rename(columns={"seconds_total":"seconds"}, inplace=True)
applewatch_df = pd.merge(applewatch_df, light_seconds, on='Date of Sleep', how='left', suffixes=('_total', '_light'))
applewatch_df.rename(columns={"seconds_total":"seconds"}, inplace=True)
applewatch_df = pd.merge(applewatch_df, deep_seconds, on='Date of Sleep', how='left', suffixes=('_total', '_deep') )
applewatch_df.rename(columns={"seconds_total":"seconds"}, inplace=True)
applewatch_df = pd.merge(applewatch_df, rem_seconds, on='Date of Sleep', how='left', suffixes=('_total', '_rem'))
applewatch_df.rename(columns={"seconds_total":"seconds_count"}, inplace=True)
applewatch_df['Category'] = applewatch_df['Category'].str.lower()

applewatch_df.rename(columns={'seconds_asleep': 'asleep_seconds', 'seconds_wake': 'awake_seconds', 
                              'seconds_light':'light_seconds', 'seconds_deep':'deep_seconds', 
                              'seconds_rem':'rem_seconds', 'Date of Sleep': 'date_of_sleep', 
                              'Category':'category'}, inplace=True)

applewatch_df['time_hours'] = applewatch_df['Start Time'].dt.hour
applewatch_df['time_minutes'] = applewatch_df['Start Time'].dt.minute
applewatch_df['time_seconds'] = applewatch_df['Start Time'].dt.second

applewatch_df.drop(columns={'End Time', 'Start Time'}, inplace=True)

print("Apple Watch data is cleaned. Shape: ", applewatch_df.shape)
final_dataset = pd.concat([fitbit_df, applewatch_df])

final_dataset['category'] = final_dataset['category'].replace({'wake': 'awake'})

final_dataset['weekday'] = pd.to_datetime(final_dataset['date_of_sleep']).dt.strftime('%A')
#final_dataset = final_dataset.drop(columns='date_of_sleep')

final_dataset.to_csv("./data/final_dataset.csv", index=False)
