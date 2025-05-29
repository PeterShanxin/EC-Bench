import pandas as pd
import json
from sklearn.model_selection import train_test_split
import os 

# conver a csv file with Entry,EC number,Sequence columns to a json file in following format:
# {"id_uniprot": {"Entry1": "Entry1", "Entry2": "Entry2", ...}, "ec_number": {"Entry1": "EC1", "Entry2": "EC2", ...}, "sequence": {"Entry1": "Sequence1", "Entry2": "Sequence2", ...}}
def convert_csv_to_json(df, json_file):
    dict = {"id_uniprot": {}, "sequence": {}, "ec_number": {}}
    for index, row in df.iterrows():
        dict["id_uniprot"][row['id']] = row['id']
        dict["sequence"][row['id']] = row['seq']
        dict["ec_number"][row['id']] = row['ec_number']
    with open(json_file, 'w') as outfile:
        json.dump(dict, outfile)

# split train_ec files to train and valid files 
train_ec = pd.read_csv('data/cluster-30/train_ec.csv')
train, valid = train_test_split(train_ec, test_size=0.1, random_state=42)
test = pd.read_csv('data/test_ec.csv')
price = pd.read_csv('data/price-149.csv')
train_blastp = pd.concat([train, valid])
# shuffle train_blastp
train_blastp = train_blastp.sample(frac=1).reset_index(drop=True)

# remove rows with having more than one EC number in EC number column; having ',' in EC number column
#train = train[~train['ec_number'].str.contains(",")]
#valid = valid[~valid['ec_number'].str.contains(",")]

# expand EC number column to multiple rows with one EC number in each row: example: 1.2.3.4,5.6.7.8 -> two rows: one with 1.2.3.4 and the other with 5.6.7.8
train = train.assign(ec_number=train['ec_number'].str.split(',')).explode('ec_number')
valid = valid.assign(ec_number=valid['ec_number'].str.split(',')).explode('ec_number')

# replace EC number '-' with 0.0.0.0
#train['ec_number'] = train['ec_number'].replace('-', '0.0.0.0')
#valid['ec_number'] = valid['ec_number'].replace('-', '0.0.0.0')
#train_ec['ec_number'] = train_ec['ec_number'].replace('-', '0.0.0.0')

if not os.path.exists('tfpc/data/datasets/mine_30'):
    os.makedirs('tfpc/data/datasets/mine_30')
convert_csv_to_json(train, 'tfpc/data/datasets/mine_30/mine_30_train.json')
convert_csv_to_json(valid, 'tfpc/data/datasets/mine_30/mine_30_valid.json')
convert_csv_to_json(test, 'tfpc/data/datasets/mine_30/mine_30_test.json')
convert_csv_to_json(train_blastp, 'tfpc/data/datasets/mine_30/mine_30_train_blastp.json')
if not os.path.exists('tfpc/data/datasets/price_blastp.json'):
    convert_csv_to_json(price, 'tfpc/data/datasets/price_blastp.json')


# split train_ec files to train and valid files
train_ec = pd.read_csv('data/cluster-100/train_ec.csv')
train, valid = train_test_split(train_ec, test_size=0.1, random_state=42)
if not os.path.exists('tfpc/data/datasets/mine_100'):
    os.makedirs('tfpc/data/datasets/mine_100')
train.to_csv('tfpc/data/datasets/mine/cluster-100/train.csv', index=False)
valid.to_csv('tfpc/data/datasets/mine/cluster-100/valid.csv', index=False)

# remove rows with having more than one EC number in EC number column; having ',' in EC number column
train = train[~train['ec_number'].str.contains(",")]
valid = valid[~valid['ec_number'].str.contains(",")]

# replace EC number '-' with '0.0.0.0'
train['ec_number'] = train['ec_number'].replace('-', '0.0.0.0')
valid['ec_number'] = valid['ec_number'].replace('-', '0.0.0.0')
train_ec['ec_number'] = train_ec['ec_number'].replace('-', '0.0.0.0')

if not os.path.exists('tfpc/data/datasets/mine_100'):
    os.makedirs('tfpc/data/datasets/mine_100')
convert_csv_to_json(train, 'tfpc/data/datasets/mine_100/mine_100_train.json')
convert_csv_to_json(valid, 'tfpc/data/datasets/mine_100/mine_100_valid.json')
convert_csv_to_json(train_ec, 'tfpc/data/datasets/mine/cluster-100/train_blastp.json')


price = pd.read_csv('data/price.csv')
price = price[~price['ec_number'].str.contains(",")]

convert_csv_to_json(price, 'tfpc/data/datasets/mine/price.json')

test_ec = pd.read_csv('data/test_ec.csv')
test_ec = test_ec[~test_ec['ec_number'].str.contains(",")]
test_ec['ec_number'] = test_ec['ec_number'].replace('-', '0.0.0.0')

convert_csv_to_json(test_ec, 'tfpc/data/datasets/mine/test_ec.json')

# merge price and test_ec files
price_test = pd.concat([price, test_ec])
convert_csv_to_json(price_test, 'tfpc/data/datasets/mine/mine_test.json')
# copy mine_test.json to mine_30, mine_50, mine_70, mine_90, mine_100
import shutil
shutil.copy('tfpc/data/datasets/mine/mine_test.json', 'tfpc/data/datasets/mine_30/mine_30_test.json')
shutil.copy('tfpc/data/datasets/mine/mine_test.json', 'tfpc/data/datasets/mine_50/mine_50_test.json')
shutil.copy('tfpc/data/datasets/mine/mine_test.json', 'tfpc/data/datasets/mine_70/mine_70_test.json')
shutil.copy('tfpc/data/datasets/mine/mine_test.json', 'tfpc/data/datasets/mine_90/mine_90_test.json')
shutil.copy('tfpc/data/datasets/mine/mine_test.json', 'tfpc/data/datasets/mine_100/mine_100_test.json')
