import os
import base64
import requests
from io import BytesIO
import numpy as np
import pandas as pd
import pickle

# Get OpenAI API Key from environment variable
api_key = ...
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

metaprompt = '''
- You are an expert interior designer thinking about placing objects in a room that follows instructions.
'''    

# Function to encode the image
def encode_image_from_file(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def encode_image_from_pil(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def prepare_inputs(message):
    payload = {
        "model": "gpt-4-1106-preview",
        "messages": [
        {
            "role": "system",
            "content":
                metaprompt
        }, 
        {
            "role": "user",
            "content": message,
        }
        ],
        "max_tokens": 800
    }

    return payload

def request_gpt4turbo(message):
    payload = prepare_inputs(message)
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    res = response.json()['choices'][0]['message']['content']
    return res


cap3D = pd.read_csv('../Cap3D_automated_Objaverse_no3Dword.csv', names=['uid', 'caption'])
cap3D = cap3D.iloc[:10_000].copy()

features = {}

if os.path.exists('classify_cap.pkl'):
    with open('classify_cap.pkl', 'rb') as f:
        features = pickle.load(f)
        progress_uids = list(features.keys())

for i, row in cap3D[~cap3D['uid'].isin(progress_uids)].iterrows():
    print(i, flush=True)
    uid = row['uid']
    caption = row['caption']
    classify_prompt = f"""
Given the description of this object here "{caption}", describe the basic color, material, theme, and shape through a single word for each feature. 
Return a this in the format of a string 'Color, Material, Theme, Shape'. 
If the description does not describe one of these features, describe the feature as "N/A".
Remember to describe the features as basic colors, materials, themes, and shapes.
Remember, you do not need to explain your reasoning, just output the string of 4 words seperated by commas.
"""
    description = request_gpt4turbo(classify_prompt)
    n_commas = description.count(',') == 3
    n = 0
    while not n_commas:
        description = request_gpt4turbo(classify_prompt)
        n_commas = description.count(',') == 3
        n += 1
        if n >= 10:
            description = "N/A, N/A, N/A, N/A"
            break
    features.setdefault(uid, {})
    features[uid] = description
    if i % 500:
        with open('classify_cap.pkl', 'wb') as f:
            pickle.dump(features, f)
with open('classify_cap.pkl', 'wb') as f:
    pickle.dump(features, f)