
# new auth backend, referencing main.py

## reproduce
1- create venv with python version 3.11.9
2- install requirements from requirements.txt
3- run main.py file on 8018 port
```
uvicorn main:app --host 0.0.0.0 --port 8018 --reload --log-level debug
```
4- run the following curl command to login to github
```bash

curl http://localhost:8018/login/github
```
5- get the session cookie from the response and use it in the following curl commands

6- run the following curl command to list all chats
```bash

curl -X POST http://localhost:8018/chats \
-H "Cookie: github_session=your_session_cookie"
```

returns a list of all chats available in the user session


7- run the following curl command to create a new chat
```bash

curl -X POST http://localhost:8018/create_chat \

-H "Cookie: github_session=your_session_cookie"  # This will come from GitHub auth
```

creates a new chat in the user session

Follow the below documentation for endpoints, it covers querying a chat session, deleting a chat session, getting variables and files in a chat session, and restarting a chat session and deleting a specific session or all chat sessions, and also uploading a csv file.

## Endpoints

```
curl http://localhost:8018/login/github
```
this reroutes to login to github and returns via a callback a json with the github_session cookie needed for authenticating the api calls


endpoint create a chat
```
curl -X POST http://localhost:8018/chats \
-H "Cookie: github_session=your_session_cookie"  # This will come from GitHub auth
```


endpoint to get all chats per user
```
curl http://localhost:8018/chats \
-H "Cookie: github_session=your_session_cookie"
```


endpoint to query a chat with chat id and user id

```
curl -X POST http://localhost:8018/query/chat_20250202_162145 \
-H "Content-Type: application/json" \
-H "Cookie: github_session=your_session_cookie" \
-d '{"query": "create a simple plot showing a sine wave"}'
```

endpoint to delete chats

delete specific chat
```
curl -X DELETE http://localhost:8018/chat/chat_20250131_123456 \
-H "Cookie: github_session=your_session_cookie"
```

delete all chats

```
curl -X DELETE http://localhost:8018/chats/all \
-H "Cookie: github_session=your_session_cookie"
```


endpoint to get variables in chat id

```
curl http://localhost:8018/variables/chat_20250131_123456 \
-H "Cookie: github_session=your_session_cookie"
```


get variable

```
curl http://localhost:8018/variable/chat_20250131_123456/df \
-H "Cookie: github_session=your_session_cookie"
```

endpoint to get files in chat id

```
curl http://localhost:8018/files/chat_20250131_123456 \
-H "Cookie: github_session=your_session_cookie"
```

get file

```
curl http://localhost:8018/file/chat_20250131_123456/plot.png \
-H "Cookie: github_session=your_session_cookie" \
--output plot.png
```

for text:
```
curl http://localhost:8018/file/chat_20250131_123456/data.csv \
-H "Cookie: github_session=your_session_cookie"
```


endpoint to restart the session

```
curl -X POST http://localhost:8018/restart/chat_20250202_162145 \
-H "Cookie: github_session=your_session_cookie"
```

endpoint to upload and load csv file in memory

```
curl -X POST http://localhost:8015/upload_csv/chat_20250201_182549 \
-H "Content-Type: application/json" \
-H "Cookie: github_session=your_session_cookie" \
-d '{"file_path": "/Users/JJneid/Desktop/TD project/tdi_rentals_dataset.csv"}'
```



