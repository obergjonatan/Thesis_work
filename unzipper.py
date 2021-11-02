import gzip
import shutil
import datetime as dt
from os.path import exists

base_path = 'Covid_tweet_data/covid19_twitter/dailies/'

start_date = dt.date(2020, 3, 22)
end_date = dt.date(2021, 10, 19)

date = start_date

while date <= end_date:
    path = base_path + str(date) + '/' + str(date)+'_clean-dataset.tsv.gz'
    print(path)
    new_path = base_path + str(date) + '/' + str(date)+'_clean-dataset.tsv'
    if not exists(new_path):
        with gzip.open(path, 'rb') as f_in:
            with open(new_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
        print('Already exists no need to unpack')
    date += dt.timedelta(days=1)


path = base_path + str(date) + '/' + str(date)+'_clean-dataset.tsv'
print(path)
