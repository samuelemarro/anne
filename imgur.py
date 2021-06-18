import requests
import base64

CLIENT_ID = None

def upload_from_bytes(content):
    global CLIENT_ID

    if CLIENT_ID is None:
        from getpass import getpass
        CLIENT_ID = getpass('Enter the Imgur Client ID: ')

    headers = {
    'Authorization': 'Client-ID ' + CLIENT_ID
    }

    params = {
    'image': base64.b64encode(content)
    }

    r = requests.post(f'https://api.imgur.com/3/image', headers=headers, data=params)
    print('status:', r.status_code)
    data = r.json()
    return data

def upload_from_path(path):
    return upload_from_bytes(open(path, 'rb').read())