import json
import time
import requests
import hashlib
import os


# 登录信息
username = 'cdp'
password = '123'

# 哈希密码
hashed_password = hashlib.sha256(password.encode()).hexdigest()
# tokens = []
# if os.path.exists('token.txt'):
#     with open('token.txt', 'r') as file:
#         for line in file:
#             token = json.loads(line)
#             tokens.append(token)
# flag = False
# curr_token = ''
# for token in tokens:
#     if token['username'] == username and token['password'] == hashed_password:
#
#         flag = True

url = 'http://127.0.0.1:8000/accounts/login/'
session = requests.Session()
csrf_token = session.get(url).cookies['csrftoken']

# if flag:
#     headers = {'Authorization': f'Token {token}'}
#     response = requests.get(url, headers=headers)
#     if response.status_code == 200:
#         print('使用已有token登录成功')
#     else:
#         print('使用已有token登录失败')
#
# else:

payload = {'username': username, 'password': password}
response = session.post(url, data=payload, headers={'X-CSRFToken': csrf_token})

if response.status_code == 200:

    token = response.text
    # with open('token.txt', 'a') as file:
    #
    #     file.write('\n' + token)
    #     file.flush()
    #     print("登录成功，并且 token 已保存到 token.txt 文件中。")
    #     file.close()
    print("登录成功")
else:
    print("登录失败：", response.status_code, response.text)



