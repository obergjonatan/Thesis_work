import pandas as pd
import datetime as dt
from datetime import timedelta
import csv

basestring = "https://raw.githubusercontent.com/thepanacealab/covid19_twitter/master/dailies/"

start_date = dt.date(2020, 7, 29)
search_words = [
    "lockdown", "mask", "quarantine", "close", "shutdown", "distancing",
    "gatherings"
]
days_to_check = 180
daily_matches = {
    str(start_date): 0,
}

date = start_date
for i in range(days_to_check):
    url = basestring + str(date) + "/" + str(date) + "_top1000terms.csv"
    data = pd.read_csv(url, header=None)
    data = data.fillna(0)
    matches_dict = {}
    for index, row in data.iterrows():
        if row[0] in search_words:
            matches_dict[row[0]] = row[1]
    daily_matches[str(date)] = matches_dict
    date = date + timedelta(days=1)

print(daily_matches)

csv_file = open("/measures_data/discussed_measures_" + str(start_date) + "-" +
                str(start_date + timedelta(days=days_to_check)) + ".csv",
                "w",
                newline='')
writer = csv.writer(csv_file)
header = ['date'] + search_words
writer.writerow(header)
for key in daily_matches:
    row = [key]
    for measure in search_words:
        if measure in daily_matches[key]:
            row += [daily_matches[key][measure]]
        else:
            row += ['0']
    writer.writerow(row)
csv_file.close()
