import json

json_path = 'json_data/gesture_Curtain Close.json'
data = []

with open(json_path, 'r') as jf:
    content = json.load(jf)
    samples = content['samples']
    for sample in samples:
        data.append(sample)


print(len(data))