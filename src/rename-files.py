import os

DATA_PATH = '/Users/raimibinkarim/Desktop/ENet_data'

for folder in os.listdir(DATA_PATH):
    if not os.path.isdir(os.path.join(DATA_PATH, folder)):
        continue
    for file in os.listdir(os.path.join(DATA_PATH, folder)):
        try:
            new_name = os.path.join(DATA_PATH, folder, 'Wound_' + file.split("_")[1])
            print(new_name)
            os.rename(os.path.join(DATA_PATH, folder, file), new_name)
        except IndexError:
            print("Cannot rename file")