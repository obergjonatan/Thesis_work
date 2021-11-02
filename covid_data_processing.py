import pandas as pd
import csv
import datetime as dt
from datetime import time, timedelta

basestring = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/"

start_date = dt.date(2020, 7, 29)
days_to_check = 180
confirmed_dict = {}
date = start_date-timedelta(days=8)

for i in range(days_to_check + 16):
    url = basestring + date.strftime("%m-%d-%Y") + ".csv"
    data = pd.read_csv(url)
    data = data.fillna(0)
    sum_active_cases = 0
    for index, row in data.iterrows():
        if row[3] == 'US':
            sum_active_cases += int(row[7])
    confirmed_dict[str(date)] = sum_active_cases
    date = date + timedelta(days=1)


new_cases_dict = {}
date = start_date-timedelta(days=7)
for i in range(days_to_check+15):
    new_cases_dict[str(date)] = confirmed_dict[str(date)] - \
        confirmed_dict[str(date-timedelta(days=1))]
    date = date + timedelta(days=1)

confirmed_dict = new_cases_dict


# Writes down recorded active cases of Covid 19 for each day in the specified time span
csv_file = open("new_cases_" + str(start_date) + "-" +
                str(start_date + timedelta(days=days_to_check)) + "_US.csv",
                "w",
                newline='')
writer = csv.writer(csv_file)
header = ["date", "predict_value"]
writer.writerow(header)

for i in range(days_to_check+1):
    writer.writerow([
        str(start_date + timedelta(days=i)),
        confirmed_dict[str(start_date + timedelta(days=i))]
    ])
csv_file.close()

# Writes down the 'derivative' of active cases of Covid 19 for each day in the specified time span
csv_file = open("new_cases_derivative_" + str(start_date) + "-" +
                str(start_date + timedelta(days=days_to_check)) + "_US.csv",
                "w",
                newline='')
writer = csv.writer(csv_file)
header = ["date", "predict_value"]
writer.writerow(header)
for day in range(days_to_check+1):
    change = (confirmed_dict[str(start_date + timedelta(days=day + 1))] -
              confirmed_dict[str(start_date + timedelta(days=day))]
              ) / confirmed_dict[str(start_date + timedelta(days=day))]
    writer.writerow([str(start_date + timedelta(days=day)), change])
csv_file.close()

# Writes down the 'derivative' of active cases of Covid 19 from one day to a day a week ahead for each day in the specified time span
csv_file = open("new_cases_derivative_7_days" + str(start_date) + "-" +
                str(start_date + timedelta(days=days_to_check)) + "_US.csv",
                "w",
                newline='')
writer = csv.writer(csv_file)
header = ["date", "predict_value"]
writer.writerow(header)
for day in range(days_to_check+1):
    change = (confirmed_dict[str(start_date + timedelta(days=day + 7))] -
              confirmed_dict[str(start_date + timedelta(days=day))]
              ) / confirmed_dict[str(start_date + timedelta(days=day))]
    writer.writerow([str(start_date + timedelta(days=day)), change])
csv_file.close()


# Calculates the moving_average of active cases to be used later
moving_average = {}
moving_span = 7
first_day_average = 0
for i in range(moving_span):
    first_day_average += confirmed_dict[str(
        start_date-timedelta(moving_span-i))]
first_day_average = first_day_average/moving_span

for i in range(days_to_check+8):
    moving_average[str(start_date+timedelta(days=i))] = first_day_average
    first_day_average += (confirmed_dict[str(start_date+(timedelta(days=i)))] -
                          confirmed_dict[str(start_date+(timedelta(days=i-moving_span)))])/moving_span

# Writes down the 'derivative' of trailing moving mean of active cases of Covid 19 for each day in the specified time span
csv_file = open("new_cases_derivative_trailing_moving_mean" + str(start_date) + "-" +
                str(start_date + timedelta(days=days_to_check)) + "_US.csv",
                "w",
                newline='')
writer = csv.writer(csv_file)
header = ["date", "predict_value"]
writer.writerow(header)
for day in range(days_to_check+1):
    change = (moving_average[str(start_date + timedelta(days=day + 1))] -
              moving_average[str(start_date + timedelta(days=day))]
              ) / moving_average[str(start_date + timedelta(days=day))]
    writer.writerow([str(start_date + timedelta(days=day)), change])
csv_file.close()


# Writes down the 'derivative' of trailing moving mean of active cases from one day to a day a week ahead of Covid 19 for each day in the specified time span
csv_file = open("new_cases_derivative_trailing_moving_mean_7_days" + str(start_date) + "-" +
                str(start_date + timedelta(days=days_to_check)) + "_US.csv",
                "w",
                newline='')
writer = csv.writer(csv_file)
header = ["date", "predict_value"]
writer.writerow(header)
for day in range(days_to_check+1):
    change = (moving_average[str(start_date + timedelta(days=day + 7))] -
              moving_average[str(start_date + timedelta(days=day))]
              ) / moving_average[str(start_date + timedelta(days=day))]
    writer.writerow([str(start_date + timedelta(days=day)), change])
csv_file.close()
