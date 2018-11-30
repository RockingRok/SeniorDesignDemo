from flask import send_file, Flask, current_app, request, jsonify
import json
import requests

request = {'data' : 'D:/design/videos/jumpingjacks.mp4',
           'key' : 'D:/design/videos/jumpingjacks.mp4'}

url = "http://127.0.0.1:5000"
response = requests.post(url, json=request)

resp_dict = response.json()
print(json.dumps(resp_dict, indent=4, sort_keys=True))
