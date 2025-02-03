
# AI Data Scientist backend

## Intro

This is a backend service that allows users to create chat sessions and interact with the chat sessions to query, delete, get variables, get files, restart the session, and upload a csv file. 

The chat sessions are stored in memory and are NOT deleted when the server is restarted. Unless, the user deletes the chat session or we call delete all chats endpoint and file directories.

The chat sessions are created with a unique id and a timestamp. The chat sessions are stored in a dictionary with the chat id as the key and the chat session as the value. The chat session is a class that has a chat id, a user id, a list of variables, and a list of files. 

The chat session class has methods to add variables, get variables, add files, get files, and delete files. 

The chat session class also has a method to restart the session which deletes all variables and files in the chat session. 

## Front end

The goal is to have a front end that allows users to interact with the backend service. The front end will allow users to create chat sessions, query chat sessions, delete chat sessions, get variables, get files, restart the session, and upload a csv file.

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
curl -X POST http://localhost:8018/create_chat \
-H "Cookie: github_session=your_session_cookie"  # This will come from GitHub auth
```


endpoint to get all chats per user
```
curl http://localhost:8018/chats \
-H "Cookie: github_session=eyJ1c2VyIjogIkpKbmVpZCIsICJfc3RhdGVfZ2l0aHViX3JMczBIZ21zam5VcnFtQ3ppclZaaDVZN25jV1pRdyI6IHsiZGF0YSI6IHsicmVkaXJlY3RfdXJpIjogImh0dHA6Ly9sb2NhbGhvc3Q6ODAxOC9hdXRoL2dpdGh1Yi9jYWxsYmFjayIsICJ1cmwiOiAiaHR0cHM6Ly9naXRodWIuY29tL2xvZ2luL29hdXRoL2F1dGhvcml6ZT9yZXNwb25zZV90eXBlPWNvZGUmY2xpZW50X2lkPU92MjNsaWV6a0NpYlM3Q0RMZUNSJnJlZGlyZWN0X3VyaT1odHRwJTNBJTJGJTJGbG9jYWxob3N0JTNBODAxOCUyRmF1dGglMkZnaXRodWIlMkZjYWxsYmFjayZzY29wZT11c2VyJTNBZW1haWwmc3RhdGU9ckxzMEhnbXNqblVycW1DemlyVlpoNVk3bmNXWlF3In0sICJleHAiOiAxNzM4NjA2MzA5Ljg1MjU3NX19.Z6D41Q.9V7c4ZOiMecFBfFn4O45Krm79as"
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





curl http://localhost:8018/chats \     
-H "Cookie: github_session=eyJfc3RhdGVfZ2l0aHViX2JKdm81aW5NeFRmTGY2RlZGUTRjUnVXYzRNNUV3SSI6IHsiZGF0YSI6IHsicmVkaXJlY3RfdXJpIjogImh0dHA6Ly9sb2NhbGhvc3Q6ODAxOC9hdXRoL2dpdGh1Yi9jYWxsYmFjayIsICJ1cmwiOiAiaHR0cHM6Ly9naXRodWIuY29tL2xvZ2luL29hdXRoL2F1dGhvcml6ZT9yZXNwb25zZV90eXBlPWNvZGUmY2xpZW50X2lkPU92MjNsaWV6a0NpYlM3Q0RMZUNSJnJlZGlyZWN0X3VyaT1odHRwJTNBJTJGJTJGbG9jYWxob3N0JTNBODAxOCUyRmF1dGglMkZnaXRodWIlMkZjYWxsYmFjayZzY29wZT11c2VyJTNBZW1haWwmc3RhdGU9Ykp2bzVpbk14VGZMZjZGVkZRNGNSdVdjNE01RXdJIn0sICJleHAiOiAxNzM4NjA2MTY4LjgyNDMzNTh9fQ==.Z6D4SA.VgjomnzLJ8ULWbasxItIaYN0tLs