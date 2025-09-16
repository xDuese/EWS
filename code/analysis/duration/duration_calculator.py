import os
import re
import pandas as pd

# Folder where your CSV files are stored
data_folder = "EWS/data/raw/raw_bin"

words_per_image_dict = {
    11:7,
    12:34,
    13:15,
    15:6,
    16:19,
    17:12,
    33:25,
    74:3,
    76:2,
    82:2,
    87:6,
    88:26,
    92:5,
    99:2,
    100:11,
    102:4,
    106:12,
    107:6,
    108:27,
    109:7,
    110:12,
    111:5,
    112:9,  
    113:22,
    114:3,
    115:7,
    116:17,
    117:6,
    118:23,
    119:16,
    120:26,
    121:11,
    122:7,
    123:4,
    124:14,
    125:17,
    126:24,
    127:5,
    128:19,
    129:10,
    130:25,
    131:16,
    132:12,
    133:10,
    134:24,
    135:3,
    136:25,
    137:4,
    138:23,
    139:14,
    140:22,
    141:45,
    142:14,
    143:19,
    144:74,
    145:39,
    146:73,
    147:19,
    148:36,
    149:34,
    150:52,
    151:50,
    152:47
    }
# Regex to extract IDs from filenames
pattern = re.compile(r"P(\d{2})_IMG(\d{3})_(\d{5})\.csv")

records = []

for filename in os.listdir(data_folder):
    match = pattern.match(filename)
    if match:
        participant_id, image_id, category_code = match.groups()
        
        file_path = os.path.join(data_folder, filename)
        df = pd.read_csv(file_path)
        
        if "system_time_stamp" in df.columns and not df.empty:
            start_time = df["system_time_stamp"].iloc[0]
            end_time = df["system_time_stamp"].iloc[-1]
            duration = (end_time - start_time) / 1_000_000  # convert microseconds → seconds
            if duration < 5:
                duration = 5
                print(f"Negative duration in file {filename}, setting to 5.")
            elif duration > 15:
                duration = 15
                print(f"Excessive duration in file {filename}, setting to 15.")
        else:
            duration = None  # in case of missing data
        
        records.append({
            "participantID": participant_id,
            "imageID": image_id,
            "categoryCode": category_code,
            "duration": duration,
            "words": words_per_image_dict.get(int(image_id), 0)
        })






result_df = pd.DataFrame(records)
result_df.to_csv("EWS/code/analysis/duration/view_durations.csv", index=False)
