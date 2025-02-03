
# new auth backend, referencing main.py


auth endpoint for github
```python
oauth = OAuth()
oauth.register(
    name='github',
    client_id='your-github-client-id',
    client_secret='your-github-client-secret',
    access_token_url='https://github.com/login/oauth/access_token',
    access_token_params=None,
    authorize_url='https://github.com/login/oauth/authorize',
    authorize_params=None,
    api_base_url='https://api.github.com/',
    client_kwargs={'scope': 'user:email'},
)

def get_user_path(username: str) -> Path:
    """
    Creates and returns a user-specific directory path
    
    Args:
        username: The username (from GitHub in this case)
    
    Returns:
        Path: Path object pointing to user's directory
    """
    # Base path is your main directory (the one you defined as 'path')
    user_path = Path(path) / username
    
    # Create the directory if it doesn't exist
    user_path.mkdir(parents=True, exist_ok=True)
    
    return user_path

def get_current_user(request: Request):
    user = request.session.get('user')
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


@app.get('/login/github')
async def github_login(request: Request):
    redirect_uri = request.url_for('auth_github')
    return await oauth.github.authorize_redirect(request, redirect_uri)

@app.get('/auth/github/callback')
async def auth_github(request: Request):
    token = await oauth.github.authorize_access_token(request)
    resp = await oauth.github.get('user', token=token)
    user = resp.json()
    
    # Create user directory
    user_path = get_user_path(user['login'])
    
    # Store user info in session
    request.session['user'] = user['login']
    
    return RedirectResponse(url='/dashboard')

```

endpoint create a chat
```
curl -X POST http://localhost:8015/create_chat \
-H "Cookie: session=eyJ1c2VyIjogIkpKbmVpZCJ9.Z56oeg.ckhNPVJqdcSaHoiPRYtli8BJmWg"  # This will come from GitHub auth
```
session cookie: eyJ1c2VyIjogIkpKbmVpZCJ9.Z56oeg.ckhNPVJqdcSaHoiPRYtli8BJmWg

endpoint to get all chats per user
```
curl http://localhost:8015/chats \
-H "Cookie: session=your_session_cookie"
```

endpoint to query a chat with chat id and user id

```
curl -X POST http://localhost:8015/query/chat_20250201_182549 \
-H "Content-Type: application/json" \
-H "Cookie: session=eyJ1c2VyIjogIkpKbmVpZCJ9.Z56oeg.ckhNPVJqdcSaHoiPRYtli8BJmWg" \
-d '{"query": "create a simple plot showing a sine wave"}'
```

endpoint to delete chats

delete specific chat
```
curl -X DELETE http://localhost:8018/chat/chat_20250131_123456 \
-H "Cookie: session=your_session_cookie"
```

delete all chats

```
curl -X DELETE http://localhost:8018/chats/all \
-H "Cookie: session=your_session_cookie"
```


endpoint to get variables in chat id

```
curl http://localhost:8018/variables/chat_20250131_123456 \
-H "Cookie: session=your_session_cookie"
```


get variable

```
curl http://localhost:8018/variable/chat_20250131_123456/df \
-H "Cookie: session=your_session_cookie"
```

endpoint to get files in chat id

```
curl http://localhost:8018/files/chat_20250131_123456 \
-H "Cookie: session=your_session_cookie"
```

get file

```
curl http://localhost:8018/file/chat_20250131_123456/plot.png \
-H "Cookie: session=your_session_cookie" \
--output plot.png
```

for text:
```
curl http://localhost:8018/file/chat_20250131_123456/data.csv \
-H "Cookie: session=your_session_cookie"
```


endpoint to restart the session

```
curl -X POST http://localhost:8018/restart/chat_20250131_123456 \
-H "Cookie: session=your_session_cookie"
```

endpoint to upload and load csv file in memory

```
curl -X POST http://localhost:8015/upload_csv/chat_20250201_182549 \
-H "Content-Type: application/json" \
-H "Cookie: session=eyJ1c2VyIjogIkpKbmVpZCJ9.Z56oeg.ckhNPVJqdcSaHoiPRYtli8BJmWg" \
-d '{"file_path": "/Users/JJneid/Desktop/TD project/tdi_rentals_dataset.csv"}'
```
curl http://localhost:8015/variables/chat_20250201_182549 \
-H "Cookie: session=eyJ1c2VyIjogIkpKbmVpZCJ9.Z56oeg.ckhNPVJqdcSaHoiPRYtli8BJmWg"


# New chat id backend referencing main_chat.py



we separate by chatID and create directory by chat ID, the agent handles the referencing of variables and files


endpoint create chat
```
curl -X POST http://localhost:8018/create_chat \
-H "Content-Type: application/json"
```

endpoint to list chats
```
curl http://localhost:8018/chats
```


endpoint query chat id
```
curl -X POST http://localhost:8018/query/chat_123 \
-H "Content-Type: application/json" \
-d '{"query": "analyze the last 5 days of AAPL stock data"}'
```

endpoint to get variabels in chat id
```
curl -X POST http://localhost:8018/query/chat_123 \
-H "Content-Type: application/json" \
-d '{"query": "analyze the last 5 days of AAPL stock data"}'
```

```
curl http://localhost:8018/variable/chat_123/df
```

endpoin for getting files

```
curl http://localhost:8018/file/chat_123/analysis.txt
```

```
curl http://localhost:8018/files/chat_123
```

endoint to restart the session
```
curl -X POST http://localhost:8018/restart/chat_123
```

endpoint to upload and load csv file in meory

```
curl -X POST http://localhost:8018/upload_csv/chat_123 \
-H "Content-Type: application/json" \
-d '{"file_path": "/path/to/your/file.csv"}'
```



# testing backend

curl -X POST http://localhost:8018/query/chat_20250201_165704 \
-H "Content-Type: application/json" \
-d '{"query": "analyze the last 5 days of AAPL stock data"}'

curl http://localhost:8014/variables/chat_20250201_165704

curl http://localhost:8014/files/chat_20250201_165704


curl -X POST http://localhost:8014/restart/chat_20250201_165704

# Old Basic Backend referecning main_basic.py

it has 5 endpoints in main.py
![image_name](assets/demo.png)

![image_name](assets/demo_2.png)

query
```
curl -X POST http://localhost:8012/query \
-H "Content-Type: application/json" \
-d '{"query": "analyze the last 5 days of AAPL stock data"}'
```

variables

```
curl http://localhost:8012/variables
```

```
curl http://localhost:8012/variable/apple_stock_data
```


files

```
curl http://localhost:8012/files
```

```
curl -O http://localhost:8012/file/plot.png
```

```
restart, restas the session and delete all files generated in the direcotry

curl -X POST http://localhost:8017/restart
```

```
upload csv file and load it i mmemory

curl -X POST http://localhost:8012/upload_csv \
-H "Content-Type: application/json" \
-d '{"file_path": "/Users/JJneid/Desktop/TD project/tdi_rentals_dataset.csv"}'
```


main is a backend session, i still need to design and implement the concept of sessions to create new chats and new directory for each chat, this is tricky because i need a new set of endpoints on a different URL based on the current design i did

also the concept of new user? 

# Front end

it has basic components

chat

show variables

show files


# Issues


Access issues: "It appears that there continues to be an issue with downloading files and generating plots in this environment. As such, I cannot execute the tasks of saving the Titanic dataset or plotting the age distribution directly here."
>>>> this has been solved

dependnecies within the jupyter server
>>> currently patched, can be handled better

jupyter server running locally

# next

Add a cleanup restart endpoint

introduce the concept of session


add a session token and validate with a database oyu create to keep track


uvicorn main:app --reload --port 8012 --host 127.0.0.1

frontend 

