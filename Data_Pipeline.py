import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import time

start_time = time.time()

fitbit_dataset1 = pd.read_json("./data/fitbit_sleep_data_1.json")
fitbit_dataset2 = pd.read_json("./data/fitbit_sleep_data_2.json")
fitbit_dataset3 = pd.read_json("./data/fitbit_sleep_data_3.json")
fitbit_dataset = pd.json_normalize(
    pd.concat(
        [fitbit_dataset3, fitbit_dataset1, fitbit_dataset2], ignore_index=True
    ).to_dict("records")
)

selected_columns = [
    "sleep.dateOfSleep",
    "sleep.minutesAsleep",
    "sleep.minutesAwake",
    "sleep.levels.data",
    "sleep.levels.summary.deep.minutes",
    "sleep.levels.summary.light.minutes",
    "sleep.levels.summary.rem.minutes",
]

selected_data = fitbit_dataset[selected_columns].copy()

flattened_data = []
for index, row in selected_data.iterrows():
    for item in row["sleep.levels.data"]:
        date_time = datetime.strptime(item["dateTime"], "%Y-%m-%dT%H:%M:%S.%f")
        flattened_row = {
            "date_of_sleep": row["sleep.dateOfSleep"],
            "asleep_seconds": row["sleep.minutesAsleep"] * 60,
            "awake_seconds": row["sleep.minutesAwake"] * 60,
            "category": item["level"],
            "time_hours": date_time.hour,
            "time_minutes": date_time.minute,
            "time_seconds": date_time.second,
            "seconds_count": item["seconds"],
            "deep_seconds": row["sleep.levels.summary.deep.minutes"] * 60,
            "light_seconds": row["sleep.levels.summary.light.minutes"] * 60,
            "rem_seconds": row["sleep.levels.summary.rem.minutes"] * 60,
        }
        flattened_data.append(flattened_row)

fitbit_df = pd.DataFrame(flattened_data)
fitbit_df["source"] = "fitbit"
fitbit_df.dropna(inplace=True)
print("Fitbit data is cleaned. Shape: ", fitbit_df.shape)

applewatch_df = pd.read_csv("./data/apple_watch_sleep_data.csv")
applewatch_df["Start Time"] = pd.to_datetime(applewatch_df["Start Time"])
applewatch_df["Date of Sleep"] = applewatch_df["Start Time"].dt.date

seconds_asleep = (
    applewatch_df.loc[applewatch_df["Category"].isin(["Light", "Deep", "REM"])]
    .groupby("Date of Sleep")["seconds"]
    .sum()
    .reset_index()
)
seconds_awake = (
    applewatch_df.loc[applewatch_df["Category"] == "Awake"]
    .groupby("Date of Sleep")["seconds"]
    .sum()
    .reset_index()
)
light_seconds = (
    applewatch_df.loc[applewatch_df["Category"] == "Light"]
    .groupby("Date of Sleep")["seconds"]
    .sum()
    .reset_index()
)
deep_seconds = (
    applewatch_df.loc[applewatch_df["Category"] == "Deep"]
    .groupby("Date of Sleep")["seconds"]
    .sum()
    .reset_index()
)
rem_seconds = (
    applewatch_df.loc[applewatch_df["Category"] == "REM"]
    .groupby("Date of Sleep")["seconds"]
    .sum()
    .reset_index()
)

applewatch_df = applewatch_df[applewatch_df["Category"] != "Unspecified"]

applewatch_df = pd.merge(
    applewatch_df,
    seconds_asleep,
    on="Date of Sleep",
    how="left",
    suffixes=("_total", "_asleep"),
)
applewatch_df.rename(columns={"seconds_total": "seconds"}, inplace=True)
applewatch_df = pd.merge(
    applewatch_df,
    seconds_awake,
    on="Date of Sleep",
    how="left",
    suffixes=("_total", "_wake"),
)
applewatch_df.rename(columns={"seconds_total": "seconds"}, inplace=True)
applewatch_df = pd.merge(
    applewatch_df,
    light_seconds,
    on="Date of Sleep",
    how="left",
    suffixes=("_total", "_light"),
)
applewatch_df.rename(columns={"seconds_total": "seconds"}, inplace=True)
applewatch_df = pd.merge(
    applewatch_df,
    deep_seconds,
    on="Date of Sleep",
    how="left",
    suffixes=("_total", "_deep"),
)
applewatch_df.rename(columns={"seconds_total": "seconds"}, inplace=True)
applewatch_df = pd.merge(
    applewatch_df,
    rem_seconds,
    on="Date of Sleep",
    how="left",
    suffixes=("_total", "_rem"),
)
applewatch_df.rename(columns={"seconds_total": "seconds_count"}, inplace=True)
applewatch_df["Category"] = applewatch_df["Category"].str.lower()

applewatch_df.rename(
    columns={
        "seconds_asleep": "asleep_seconds",
        "seconds_wake": "awake_seconds",
        "seconds_light": "light_seconds",
        "seconds_deep": "deep_seconds",
        "seconds_rem": "rem_seconds",
        "Date of Sleep": "date_of_sleep",
        "Category": "category",
    },
    inplace=True,
)

applewatch_df["time_hours"] = applewatch_df["Start Time"].dt.hour
applewatch_df["time_minutes"] = applewatch_df["Start Time"].dt.minute
applewatch_df["time_seconds"] = applewatch_df["Start Time"].dt.second

applewatch_df.drop(columns={"End Time", "Start Time", "Heart Rate"}, inplace=True)
applewatch_df["group"] = (
    (applewatch_df["time_hours"] >= 15) & (applewatch_df["time_hours"] < 24)
).astype(int)
applewatch_df["date_of_sleep"] = np.where(
    applewatch_df["group"] == 1,
    applewatch_df["date_of_sleep"] + pd.Timedelta(days=1),
    applewatch_df["date_of_sleep"],
)
applewatch_df.drop("group", axis=1, inplace=True)
applewatch_df["source"] = "applewatch"


print("Apple Watch data is cleaned. Shape: ", applewatch_df.shape)
final_dataset = pd.concat([fitbit_df, applewatch_df])

final_dataset["category"] = final_dataset["category"].replace({"wake": "awake"})

final_dataset["weekday"] = pd.to_datetime(final_dataset["date_of_sleep"]).dt.strftime(
    "%A"
)

final_dataset["date_of_sleep"] = pd.to_datetime(final_dataset["date_of_sleep"])


def time_to_sleep_stages(group, sleep_stages):
    time_to_stages = {}

    for stage in sleep_stages:
        stage_rows = group[group["category"] == stage]
        if not stage_rows.empty:
            stage_index = stage_rows.index[0]
            seconds_before_stage = group.loc[: stage_index - 1, "seconds_count"].sum()
            time_to_stages[stage] = seconds_before_stage
        else:
            time_to_stages[stage] = None

    return pd.Series(time_to_stages)


time_to_sleep_stages_df = final_dataset.groupby(["date_of_sleep"]).apply(
    time_to_sleep_stages, sleep_stages=["deep", "rem"]
)

stage1_dataset = final_dataset.merge(
    time_to_sleep_stages_df, left_on="date_of_sleep", right_index=True
)
stage1_dataset.rename(
    columns={"deep": "time_to_deep_seconds", "rem": "time_to_rem_seconds"}, inplace=True
)

stage1_dataset["date_of_sleep"] = pd.to_datetime(stage1_dataset["date_of_sleep"])
data_f = stage1_dataset.groupby("date_of_sleep").first().reset_index()
data_l = stage1_dataset.groupby("date_of_sleep").last().reset_index()

weekly_f_sleep = []
weekly_l_sleep = []
for i in range(0, len(data_f), 7):
    chunk = data_f.iloc[i : i + 7]
    weekly_f_sleep.append(chunk)

for i in range(0, len(data_l), 7):
    chunk = data_l.iloc[i : i + 7]
    weekly_l_sleep.append(chunk)

stage_seconds = [
    [
        cat["awake_seconds"].sum(),
        cat["light_seconds"].sum(),
        cat["deep_seconds"].sum(),
        cat["rem_seconds"].sum(),
    ]
    for cat in weekly_f_sleep
]

stage_proportions = [week / sum(week) for week in stage_seconds]

st_start_time = []
st_end_time = []

for week in weekly_f_sleep:
    start_times = list()
    for _, row in week.iterrows():
        if row["time_hours"] >= 17 and row["time_hours"] <= 23:
            time_obj = datetime(
                2024, 12, 1, row["time_hours"], row["time_minutes"], row["time_seconds"]
            ).timestamp()
        else:
            time_obj = datetime(
                2024, 12, 2, row["time_hours"], row["time_minutes"], row["time_seconds"]
            ).timestamp()

        start_times.append(time_obj)

    min_time = str(datetime.fromtimestamp(min(start_times)).time())
    max_time = str(datetime.fromtimestamp(max(start_times)).time())
    avg_time = str(datetime.fromtimestamp(sum(start_times) // len(start_times)).time())
    time_range = str(datetime.fromtimestamp(max(start_times) - min(start_times)).time())

    st_start_time.append((min_time, max_time, avg_time, time_range))


for week in weekly_l_sleep:
    end_times = list()
    for _, row in week.iterrows():
        if row["time_hours"] >= 17 and row["time_hours"] <= 23:
            time_obj = datetime(
                2024, 12, 1, row["time_hours"], row["time_minutes"], row["time_seconds"]
            ).timestamp()
        else:
            time_obj = datetime(
                2024, 12, 2, row["time_hours"], row["time_minutes"], row["time_seconds"]
            ).timestamp()

        end_times.append(time_obj)

    min_time = str(datetime.fromtimestamp(min(end_times)).time())
    max_time = str(datetime.fromtimestamp(max(end_times)).time())
    avg_time = str(datetime.fromtimestamp(sum(end_times) // len(end_times)).time())
    time_range = str(datetime.fromtimestamp(max(end_times) - min(end_times)).time())

    st_end_time.append((min_time, max_time, avg_time, time_range))

time_to_sound_sleep = []

for week in weekly_f_sleep:
    time_to_sound_sleep.append(
        (
            round(week.loc[:, "time_to_deep_seconds"].mean(), 2),
            round(week.loc[:, "time_to_rem_seconds"].mean(), 2),
        )
    )

prop_sound_sleep = []

for week in weekly_f_sleep:
    deep, rem = 0, 0

    for _, row in week.iterrows():
        deep += (
            row["time_to_deep_seconds"]
            / (row["asleep_seconds"] + row["awake_seconds"])
            * 100
        )
        rem += (
            row["time_to_rem_seconds"]
            / (row["asleep_seconds"] + row["awake_seconds"])
            * 100
        )

    deep /= week.shape[0]
    rem /= week.shape[0]

    prop_sound_sleep.append((round(deep, 2), round(rem, 2)))

final_insights = {
    "week_start": [
        str(weekly_f_sleep[index].iloc[0, 0].date())
        for index in range(len(weekly_f_sleep))
    ],
    "week_end": [
        str(weekly_f_sleep[index].iloc[-1, 0].date())
        for index in range(len(weekly_f_sleep))
    ],
    "awake_proportion": [round(val[0], 2) for val in stage_proportions],
    "light_sleep_proportion": [round(val[1], 2) for val in stage_proportions],
    "deep_sleep_proportion": [round(val[2], 2) for val in stage_proportions],
    "rem_sleep_proportion": [round(val[3], 2) for val in stage_proportions],
    "start_min_time": [val[0] for val in st_start_time],
    "start_max_time": [val[1] for val in st_start_time],
    "start_avg_time": [val[2] for val in st_start_time],
    "start_deviation": [val[3] for val in st_start_time],
    "end_min_time": [val[0] for val in st_end_time],
    "end_max_time": [val[1] for val in st_end_time],
    "end_avg_time": [val[2] for val in st_end_time],
    "end_deviation": [val[3] for val in st_end_time],
    "time_to_deep_sleep": [val[0] for val in time_to_sound_sleep],
    "time_to_rem_sleep": [val[1] for val in time_to_sound_sleep],
    "deep_sleep_start_proportion": [val[0] for val in prop_sound_sleep],
    "rem_sleep_start_proportion": [val[1] for val in prop_sound_sleep],
}

final_insights = pd.DataFrame(final_insights)

final_insights["start_deviation"] = final_insights["start_deviation"].apply(
    lambda value: (
        datetime.strptime(value, "%H:%M:%S") - datetime(1900, 1, 1)
    ).total_seconds()
    / 60.0
)
final_insights["end_deviation"] = final_insights["end_deviation"].apply(
    lambda value: (
        datetime.strptime(value, "%H:%M:%S") - datetime(1900, 1, 1)
    ).total_seconds()
    / 60.0
)


def getTimeComponents(timestamp):
    timestamp_obj = datetime.strptime(str(timestamp), "%H:%M:%S")

    return (timestamp_obj.hour, timestamp_obj.minute)


results = []

for _, row in final_insights.iterrows():
    scores = np.zeros(8)

    st_hour, st_minutes = getTimeComponents(row["start_avg_time"])
    et_hour, et_minutes = getTimeComponents(row["end_avg_time"])

    if 18 <= st_hour <= 23:
        scores[0] = 1
    else:
        if st_hour == 0 and st_minutes < 30:
            scores[0] = 1
        else:
            scores[0] = 2

    if 0 <= et_hour <= 7:
        scores[1] = 1
    else:
        scores[1] = 2

    if row["start_deviation"] < 90:
        scores[2] = scores[0]
    else:
        scores[2] = 3

    if row["end_deviation"] < 90:
        scores[3] = scores[1]
    else:
        scores[3] = 3

    if np.ceil(row["time_to_deep_sleep"] / 60) < 30:
        scores[4] = np.random.choice([scores[0], scores[1]], 1)[0]
    else:
        scores[4] = 3

    if np.ceil(row["time_to_rem_sleep"] / 60) < 90:
        scores[5] = np.random.choice([scores[0], scores[1]], 1)[0]
    else:
        scores[5] = 3

    if row["awake_proportion"] < 0.07:
        scores[6] = 1
    elif 0.07 <= row["awake_proportion"] <= 0.12:
        scores[6] = 2
    else:
        scores[6] = 3

    if row["rem_sleep_proportion"] < 0.16:
        scores[7] = 3
    elif 0.16 <= row["rem_sleep_proportion"] <= 0.18:
        scores[7] = 2
    else:
        scores[7] = 1

    results.append(scores)

labels = []

for row in results:
    deviation = np.where(row == 3.0)[0]
    if len(deviation) > 2:
        labels.append("Dolphin")
    else:
        label1 = np.where(row == 1.0)[0]
        label2 = np.where(row == 2.0)[0]

        if len(label1) > len(label2):
            labels.append("Bear")
        else:
            labels.append("Wolf")

final_insights["labels"] = labels


df = final_insights.drop(
    [
        "week_start",
        "week_end",
        "start_min_time",
        "start_max_time",
        "start_avg_time",
        "end_min_time",
        "end_max_time",
        "end_avg_time",
        "time_to_deep_sleep",
        "time_to_rem_sleep",
    ],
    axis=1,
)

y = df.loc[:, "labels"]
X = df.iloc[:, :-1]

X = X.to_numpy()
y = y.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=6
)

clf = make_pipeline(StandardScaler(), SVC(gamma="auto"))
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

end_time = time.time()
print(end_time - start_time)

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

recall = recall_score(y_test, y_pred, average='weighted')
print("Recall:", recall)

precision = precision_score(y_test, y_pred, average='weighted')
print("Precision:", precision)

f1 = f1_score(y_test, y_pred, average='weighted')
print("F1 Score:", f1)

y_test_binary = label_binarize(y_test, classes=clf.classes_)
y_pred_prob = clf.decision_function(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(clf.classes_)):
    fpr[i], tpr[i], _ = roc_curve(y_test_binary[:, i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8, 6))
for i in range(len(clf.classes_)):
    plt.plot(fpr[i], tpr[i], label=f'Class {clf.classes_[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('OvR ROC Curve')
plt.legend(loc="lower right")
plt.show()

#Visualizations
import matplotlib.pyplot as plt

final_dataset['time'] = pd.to_datetime(final_dataset['date_of_sleep']) + pd.to_timedelta(final_dataset['time_hours'], unit='h') + pd.to_timedelta(final_dataset['time_minutes'], unit='m') + pd.to_timedelta(final_dataset['time_seconds'], unit='s')
final_dataset = final_dataset.sort_values(by='time')
transitions = final_dataset['category'].shift() != final_dataset['category']

stage_mapping = {'awake': 0, 'light': 1, 'deep': 2, 'rem': 3}
final_dataset['stage_numeric'] = final_dataset['category'].map(stage_mapping)
final_dataset['transition'] = final_dataset['stage_numeric'].diff() != 0

last_3_days = final_dataset[final_dataset['date_of_sleep'] >= final_dataset['date_of_sleep'].max() - pd.DateOffset(days=3)]

# Plot line chart
plt.figure(figsize=(10, 6))
plt.plot(last_3_days['time'], last_3_days['stage_numeric'], marker='o', linestyle='-', color='blue')
plt.title('Sleep Stage Transitions Over Time (Latest 3 days)')
plt.xlabel('Time')
plt.ylabel('Sleep Stage')
plt.yticks(range(len(stage_mapping)), stage_mapping.keys())
plt.grid(True)
plt.show()

last_4_weeks = final_dataset[final_dataset['date_of_sleep'] >= final_dataset['date_of_sleep'].max() - pd.DateOffset(weeks=4)]
weekly_stage_duration = last_4_weeks.groupby([pd.Grouper(key='date_of_sleep', freq='W-MON'), 'category'])['seconds_count'].sum().unstack(fill_value=0)

# Plot stacked bar chart
plt.figure(figsize=(12, 6))
weekly_stage_duration.plot(kind='bar', stacked=True, ax=plt.gca())
plt.title('Total Time Spent in Each Sleep Stage for Last 4 Weeks')
plt.xlabel('Week')
plt.ylabel('Total Time (seconds)')
plt.xticks(rotation=45)
plt.legend(title='Sleep Stage')
plt.show()