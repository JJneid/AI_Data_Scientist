from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from pydantic import BaseModel
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv
import datetime
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.code_execution import PythonCodeExecutionTool
from autogen_ext.code_executors.jupyter import JupyterCodeExecutor
from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock
from autogen_core.tools import FunctionTool
import pandas as pd
import pandas as pd
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from autogen_ext.tools.langchain import LangChainToolAdapter

from fastapi import FastAPI, Request
from authlib.integrations.starlette_client import OAuth
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import RedirectResponse
from fastapi import FastAPI, Request
from starlette.responses import RedirectResponse
import requests
from starlette.requests import Request

from fastapi import Request, Response
from fastapi.responses import RedirectResponse, JSONResponse
import os

from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth


from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import RedirectResponse, JSONResponse
import requests
import os



load_dotenv()



path= "/Users/JJneid/Desktop/SlashMl/AI_Data_Scientist/ui/coding"
class GlobalState:
    def __init__(self):
        self.executor = None
        self.agent = None
        self.output_dir = Path(path)

state = GlobalState()

class QueryRequest(BaseModel):
    query: str

from fastapi.middleware.cors import CORSMiddleware

async def save_files(file_name: str, content: str, file_type: str = 'txt', chat_id: str = None) -> str:
    """
    Save content to a file with proper formatting and naming convention.
    
    Args:
        file_name (str): Base name for the file (will be cleaned and formatted)
        content (str): Content to save
        file_type (str): File extension (default: 'txt')
        chat_id (str): Chat ID to determine save directory
    
    Returns:
        str: Path to saved file
    """
    try:
        # Determine save directory based on chat_id
        if chat_id:
            save_dir = state.output_dir  # Uses the current chat directory from state
        else:
            save_dir = Path(path)  # Fallback to base path
        
        # Clean the file name: remove spaces and special characters
        clean_name = "".join(c for c in file_name if c.isalnum() or c in ['_', '-'])
        clean_name = clean_name.lower()
        
        # Construct full file name
        full_name = f"{clean_name}.{file_type}"
        file_path = save_dir / full_name
        
        # Ensure directory exists
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Handle different file types
        if file_type == 'csv':
            if isinstance(content, pd.DataFrame):
                content.to_csv(file_path, index=False)
            elif isinstance(content, str):
                with open(file_path, 'w') as f:
                    f.write(content)
            elif isinstance(content, (list, dict)):
                df = pd.DataFrame(content)
                df.to_csv(file_path, index=False)
            else:
                raise ValueError(f"Unsupported CSV content type: {type(content)}")
                
        elif file_type in ['png', 'jpg', 'jpeg']:
            with open(file_path, 'wb') as f:
                f.write(content if isinstance(content, bytes) else content.encode())
        
        else:
            with open(file_path, 'w') as f:
                f.write(str(content))
        
        print(f"File saved successfully at: {file_path}")
        return str(file_path)
    
    except Exception as e:
        print(f"Error saving file: {str(e)}")
        raise

# Update the tool with new parameter
save_tool = FunctionTool(
    save_files, 
    description="save to a file in the current chat directory or base directory if no chat specified",
    name="save_tool"  # Explicitly set the name
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

from fastapi.responses import RedirectResponse

def get_current_user(request: Request) -> str:
    """Get current user from session"""
    print("Session contents:", dict(request.session))  # Debug log
    user = request.session.get('user')
    print("Found user:", user)  # Debug log
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated"
        )
    return user



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        state.executor = JupyterCodeExecutor(

            timeout=120,
            output_dir=Path(path)
        )
        await state.executor.__aenter__()
        
        tool = PythonCodeExecutionTool(state.executor)
        state.agent = AssistantAgent(
            "assistant",
            OpenAIChatCompletionClient(model="gpt-4o-mini"),
            tools=[tool,save_tool],
            reflect_on_tool_use=True,
            system_message="""
            You are a data analysis assistant. Maintain state between queries by:
            1. First check if required variables exist before recreating them
            2. Reference previously created variables and dataframes when relevant
            3. Let the user know what data is currently available in the session
            4. When creating new analysis, build upon existing data when possible

            Before every import, include code to install dependency use the following format:
            import subprocess
            subprocess.check_call(['pip', 'install', 'package_name'])


            IMportant:
            When asked to save and or download files:
            1- save the files as variables in the environemnt
            2- use the tool `save_tool`, directories to save files must be absolutely respected


            """
        )
        yield
    finally:
        # Shutdown
        if state.executor:
            await state.executor.__aexit__(None, None, None)


from starlette.config import Config

# Remove all existing OAuth and session middleware configurations and replace with:

# Initialize OAuth with config
config = Config('.env')  # This will use environment variables
oauth = OAuth()

GITHUB_CLIENT_ID = "Ov23liezkCibS7CDLeCR"
GITHUB_CLIENT_SECRET = "944b3dc9dd8e6b817ae0cbb7a031f8c56d9d6d5b"

oauth.register(
    name='github',
    client_id=GITHUB_CLIENT_ID,
    client_secret=GITHUB_CLIENT_SECRET,
    access_token_url='https://github.com/login/oauth/access_token',
    access_token_params=None,
    authorize_url='https://github.com/login/oauth/authorize',
    authorize_params=None,
    api_base_url='https://api.github.com/',
    client_kwargs={'scope': 'user:email'},
)
SECRET_KEY = os.getenv("SESSION_SECRET")



NEXTJS_UI_URL = "http://localhost:3000" 

app = FastAPI(lifespan=lifespan)

from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware


app.add_middleware(
    CORSMiddleware,  # 🔹 CORS should be added separately
    allow_origins=["http://localhost:3000"],  # ✅ Allow frontend
    allow_credentials=True,
    allow_methods=["*"],  # ✅ Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # ✅ Allow all headers
)


app.add_middleware(
    SessionMiddleware, 
    secret_key=SECRET_KEY,
    session_cookie="github_session",  # Specific name for the session cookie
    max_age=3600,  # Session duration in seconds
    same_site='lax',
    https_only=False  # Set to True in production
)

############################ endpoints ########################################

@app.get('/login/github')
async def github_login(request: Request):
    """Handle GitHub login"""
    try:
        # Get the callback URL
        redirect_uri = request.url_for('auth_github')
        print(f"Redirect URI: {redirect_uri}")  # Debug log
        
        return await oauth.github.authorize_redirect(request, redirect_uri)
    except Exception as e:
        print(f"Login error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Login failed: {str(e)}"}
        )

@app.get('/auth/github/callback', name='auth_github')
async def auth_github(request: Request):
    """Handle GitHub callback"""
    try:
        token = await oauth.github.authorize_access_token(request)
        resp = await oauth.github.get('user', token=token)
        user = resp.json()
        
        # Store user info in session
        request.session['user'] = user['login']
        
        # Create user directory
        user_path = get_user_path(user['login'])
        
        # Get session cookie from request headers
        cookies = request.cookies
        session_cookie = cookies.get('session')
        
        # Create response
        response = JSONResponse({
            "message": "Successfully logged in",
            "user": user['login'],
            "user_path": str(user_path),
            "session_data": {
                "user": request.session.get('user')
            },
            "session_cookie": session_cookie,  # Include the session cookie in response
            "cookies": dict(cookies)  # Include all cookies for debugging
        })
        
        return response
        
    except Exception as e:
        print(f"Callback error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Auth failed: {str(e)}"}
        )

# @app.get('/auth/github/callback', name='auth_github')
# async def auth_github(request: Request):
#     """Handle GitHub OAuth callback"""
#     try:
#         token = await oauth.github.authorize_access_token(request)
#         resp = await oauth.github.get('user', token=token)
#         user = resp.json()

#         # Store user info in session
#         request.session['user'] = user['login']

#         # Get all cookies from the request
#         cookies = request.cookies

#         print("🔹 Cookies from request:", cookies)  # Debugging cookies
#         print("🔹 Session stored in request:", request.session)  # Debugging session

#         # Redirect to frontend
#         return RedirectResponse(url="http://localhost:3000")

#     except Exception as e:
#         print(f"Callback error: {str(e)}")
#         return JSONResponse(status_code=500, content={"error": f"Auth failed: {str(e)}"})




class ChatSession(BaseModel):
    chat_id: str
    directory: str

@app.post("/create_chat")
async def create_chat(request: Request):
    """Create a new chat session with its own directory"""
    try:
        # Get user from session
        user = get_current_user(request)
        
        # Generate unique chat ID with timestamp
        chat_id = f"chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create user and chat specific directory
        user_path = get_user_path(user)
        chat_dir = user_path / chat_id
        chat_dir.mkdir(parents=True, exist_ok=True)
        
        # Update state output directory
        state.output_dir = chat_dir
        
        return JSONResponse(
            content={
                "message": "Chat created successfully",
                "chat_id": chat_id,
                "directory": str(chat_dir),
                "user": user
            },
            headers={
                "Access-Control-Allow-Origin": "http://localhost:3000/",
                "Access-Control-Allow-Credentials": "true",
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Failed to create chat: {str(e)}"
            },
            headers={
                "Access-Control-Allow-Origin": "http://localhost:3000/",
                "Access-Control-Allow-Credentials": "true",
            }
        )


@app.get("/chats")
async def list_chats(request: Request):
    """List all chat sessions for the current user"""
    try:
        # Get user from session
        user = get_current_user(request)
        
        # Get user's directory
        user_path = get_user_path(user)
        
        # Get all chat directories
        chats = []
        if user_path.exists():
            for chat_dir in user_path.iterdir():
                if chat_dir.is_dir() and chat_dir.name.startswith('chat_'):
                    # Get creation time from the directory
                    creation_time = datetime.datetime.fromtimestamp(chat_dir.stat().st_ctime)
                    
                    # Get last modified time
                    modified_time = datetime.datetime.fromtimestamp(chat_dir.stat().st_mtime)
                    
                    # Get number of files in chat
                    files_count = len(list(chat_dir.glob('*')))
                    
                    chats.append({
                        "chat_id": chat_dir.name,
                        "created_at": creation_time.isoformat(),
                        "last_modified": modified_time.isoformat(),
                        "files_count": files_count,
                        "path": str(chat_dir)
                    })
                    
        # Sort chats by creation time, newest first
        chats.sort(key=lambda x: x["created_at"], reverse=True)
        
        return JSONResponse(
            content={
                "chats": chats,
                "user": user,
                "total_chats": len(chats)
            },
            headers={
                "Access-Control-Allow-Origin": "http://localhost:3000/",
                "Access-Control-Allow-Credentials": "true",
            }
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to list chats: {str(e)}"},
            headers={
                "Access-Control-Allow-Origin": "http://localhost:3000/",
                "Access-Control-Allow-Credentials": "true",
            }
        )
    

@app.post("/query/{chat_id}")
async def query_agent_with_chat(chat_id: str, request: QueryRequest, request_obj: Request):
    """Process a query for a specific chat session"""
    if not state.agent or not state.executor:
        return JSONResponse(
            status_code=500,
            content={"error": "Agent or executor not initialized"}
        )
    
    try:
        # Get user from session
        user = get_current_user(request_obj)
        
        # Get user and chat specific directory
        user_path = get_user_path(user)
        chat_dir = user_path / chat_id
        
        # Verify chat exists and belongs to user
        if not chat_dir.exists():
            return JSONResponse(
                status_code=404,
                content={"error": f"Chat {chat_id} not found for user {user}"}
            )
        
        # Update state output directory to this chat
        state.output_dir = chat_dir
        
        # Process the query using Console and run_stream
        result = await Console(
            state.agent.run_stream(task=request.query)
        )
        
        # Get the final response content
        response_content = result.messages[-1].content if result.messages else ""
        
        return JSONResponse(
            content={
                "response": response_content,
                "chat_id": chat_id,
                "user": user,
                "chat_dir": str(chat_dir)
            },
            headers={
                "Access-Control-Allow-Origin": "http://localhost:3000/",
                "Access-Control-Allow-Credentials": "true",
            }
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "chat_id": chat_id
            },
            headers={
                "Access-Control-Allow-Origin": "http://localhost:3000/",
                "Access-Control-Allow-Credentials": "true",
            }
        )


# Delete a specific chat
@app.delete("/chat/{chat_id}")
async def delete_chat(chat_id: str, request: Request):
    """Delete a specific chat session and its data"""
    try:
        # Get user from session
        user = get_current_user(request)
        
        # Get user and chat specific directory
        user_path = get_user_path(user)
        chat_dir = user_path / chat_id
        
        # Verify chat exists and belongs to user
        if not chat_dir.exists():
            return JSONResponse(
                status_code=404,
                content={"error": f"Chat {chat_id} not found for user {user}"}
            )
        
        # Delete the chat directory and all its contents
        shutil.rmtree(chat_dir)
        
        return JSONResponse(
            content={
                "message": f"Chat {chat_id} deleted successfully",
                "chat_id": chat_id,
                "user": user
            },
            headers={
                "Access-Control-Allow-Origin": "http://localhost:3000/",
                "Access-Control-Allow-Credentials": "true",
            }
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Failed to delete chat: {str(e)}",
                "chat_id": chat_id
            },
            headers={
                "Access-Control-Allow-Origin": "http://localhost:3000/",
                "Access-Control-Allow-Credentials": "true",
            }
        )

# Delete all chats
@app.delete("/chats/all")
async def delete_all_chats(request: Request):
    """Delete all chat sessions for the current user"""
    try:
        # Get user from session
        user = get_current_user(request)
        
        # Get user directory
        user_path = get_user_path(user)
        
        # Count chats before deletion
        chat_count = len([d for d in user_path.iterdir() if d.is_dir() and d.name.startswith('chat_')])
        
        # Delete all chat directories
        for chat_dir in user_path.iterdir():
            if chat_dir.is_dir() and chat_dir.name.startswith('chat_'):
                shutil.rmtree(chat_dir)
        
        return JSONResponse(
            content={
                "message": "All chats deleted successfully",
                "user": user,
                "chats_deleted": chat_count
            },
            headers={
                "Access-Control-Allow-Origin": "http://localhost:3000/",
                "Access-Control-Allow-Credentials": "true",
            }
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Failed to delete chats: {str(e)}"
            },
            headers={
                "Access-Control-Allow-Origin": "http://localhost:3000/",
                "Access-Control-Allow-Credentials": "true",
            }
        )

@app.get("/variables/{chat_id}")
async def get_variables(chat_id: str, request: Request):
    """Get list of available variables in the kernel for specific chat"""
    if not state.executor:
        return {"variables": [], "chat_id": chat_id}
    
    try:
        # Get user from session
        user = get_current_user(request)
        
        # Get user and chat specific directory
        user_path = get_user_path(user)
        chat_dir = user_path / chat_id
        
        # Verify chat exists and belongs to user
        if not chat_dir.exists():
            return JSONResponse(
                status_code=404,
                content={"error": f"Chat {chat_id} not found for user {user}"}
            )
        
        # Update state output directory
        state.output_dir = chat_dir
        
        cancellation_token = CancellationToken()
        result = await state.executor.execute_code_blocks(
            [CodeBlock(
                code=f"""
print([var for var in locals().keys() 
       if not var.startswith('_') 
       and (var.startswith('df_') or not any(c.startswith('df_') for c in locals().keys()))])
                """, 
                language="python"
            )],
            cancellation_token
        )
        
        # Clean up the output and convert to list
        variables = result.output
        if isinstance(variables, str):
            variables = variables.strip('[]').replace("'", "").split(', ')
            variables = [v.strip() for v in variables if v.strip()]
        
        return JSONResponse(
            content={
                "variables": variables,
                "chat_id": chat_id,
                "chat_dir": str(chat_dir),
                "user": user
            },
            headers={
                "Access-Control-Allow-Origin": "http://localhost:3000/",
                "Access-Control-Allow-Credentials": "true",
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "chat_id": chat_id
            }
        )

@app.get("/variable/{chat_id}/{var_name}")
async def get_variable_value(chat_id: str, var_name: str, request: Request):
    """Get value of a specific variable for a specific chat"""
    if not state.executor:
        raise HTTPException(status_code=500, detail="Executor not initialized")
    
    try:
        # Get user from session
        user = get_current_user(request)
        
        # Get user and chat specific directory
        user_path = get_user_path(user)
        chat_dir = user_path / chat_id
        
        # Verify chat exists and belongs to user
        if not chat_dir.exists():
            return JSONResponse(
                status_code=404,
                content={"error": f"Chat {chat_id} not found for user {user}"}
            )
        
        # Update state output directory
        state.output_dir = chat_dir
        
        cancellation_token = CancellationToken()
        result = await state.executor.execute_code_blocks(
            [CodeBlock(code=f"print({var_name})", language="python")],
            cancellation_token
        )
        
        return JSONResponse(
            content={
                "value": result.output,
                "variable_name": var_name,
                "chat_id": chat_id,
                "user": user
            },
            headers={
                "Access-Control-Allow-Origin": "http://localhost:3000/",
                "Access-Control-Allow-Credentials": "true",
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Error getting variable {var_name}: {str(e)}",
                "chat_id": chat_id
            }
        )


@app.get("/files/{chat_id}")
async def get_files(chat_id: str, request: Request):
    """Get list of files in the chat-specific output directory"""
    try:
        # Get user from session
        user = get_current_user(request)
        
        # Get user and chat specific directory
        user_path = get_user_path(user)
        chat_dir = user_path / chat_id
        
        # Verify chat exists and belongs to user
        if not chat_dir.exists():
            return JSONResponse(
                status_code=404,
                content={"error": f"Chat {chat_id} not found for user {user}"}
            )
        
        # Get all files in chat directory
        files = []
        for file_path in chat_dir.rglob("*"):
            if file_path.is_file():
                files.append({
                    "name": file_path.name,
                    "path": str(file_path.relative_to(chat_dir)),
                    "size": file_path.stat().st_size,
                    "modified": datetime.datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                    "type": file_path.suffix[1:] if file_path.suffix else "unknown"
                })

        return JSONResponse(
            content={
                "files": files,
                "chat_id": chat_id,
                "user": user,
                "total_files": len(files)
            },
            headers={
                "Access-Control-Allow-Origin": "http://localhost:3000/",
                "Access-Control-Allow-Credentials": "true",
            }
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Error getting files: {str(e)}",
                "chat_id": chat_id
            }
        )

@app.get("/file/{chat_id}/{file_path:path}")
async def get_file_content(chat_id: str, file_path: str, request: Request):
    """Get content of a specific file from a specific chat"""
    try:
        # Get user from session
        user = get_current_user(request)
        
        # Get user and chat specific directory
        user_path = get_user_path(user)
        chat_dir = user_path / chat_id
        full_path = chat_dir / file_path
        
        # Verify chat and file exist
        if not chat_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Chat {chat_id} not found for user {user}"
            )
            
        if not full_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"File {file_path} not found in chat {chat_id}"
            )
        
        # Verify file is within chat directory (security check)
        if not str(full_path.resolve()).startswith(str(chat_dir.resolve())):
            raise HTTPException(
                status_code=403,
                detail="Access to file outside chat directory is forbidden"
            )
        
        # Handle different file types
        if full_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            return FileResponse(
                full_path,
                headers={
                    "Access-Control-Allow-Origin": "http://localhost:3000/",
                    "Access-Control-Allow-Credentials": "true",
                }
            )
        else:
            # For text files
            content = full_path.read_text()
            return JSONResponse(
                content={
                    "content": content,
                    "file_name": full_path.name,
                    "chat_id": chat_id,
                    "user": user
                },
                headers={
                    "Access-Control-Allow-Origin": "http://localhost:3000/",
                    "Access-Control-Allow-Credentials": "true",
                }
            )
            
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Error reading file: {str(e)}",
                "chat_id": chat_id,
                "file_path": file_path
            }
        )

import shutil  # Add this to your imports at the top

@app.post("/restart/{chat_id}")
async def restart_kernel(chat_id: str, request: Request):
    """Restart the Jupyter kernel and clear the specific chat directory"""
    if not state.executor:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Executor not initialized",
                "chat_id": chat_id
            }
        )
    
    try:
        # Get user from session
        user = get_current_user(request)
        
        # Get user and chat specific directory
        user_path = get_user_path(user)
        chat_dir = user_path / chat_id
        
        # Verify chat exists and belongs to user
        if not chat_dir.exists():
            return JSONResponse(
                status_code=404,
                content={"error": f"Chat {chat_id} not found for user {user}"}
            )
        
        # Clear files in the chat directory
        if chat_dir.exists():
            # Remove all contents of the directory
            for item in chat_dir.iterdir():
                if item.is_file():
                    item.unlink()  # Delete file
                elif item.is_dir():
                    shutil.rmtree(item)  # Delete directory and its contents
            
        # Create/recreate the directory
        chat_dir.mkdir(exist_ok=True)
        
        # Update state output directory
        state.output_dir = chat_dir
        
        # Restart the kernel
        await state.executor.restart()
        
        return JSONResponse(
            content={
                "message": "Kernel successfully restarted and chat directory cleared",
                "chat_id": chat_id,
                "user": user,
                "chat_dir": str(chat_dir)
            },
            headers={
                "Access-Control-Allow-Origin": "http://localhost:3000/",
                "Access-Control-Allow-Credentials": "true",
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Failed to restart kernel or clear directory: {str(e)}",
                "chat_id": chat_id
            },
            headers={
                "Access-Control-Allow-Origin": "http://localhost:3000/",
                "Access-Control-Allow-Credentials": "true",
            }
        )



class CSVUploadRequest(BaseModel):
    file_path: str

@app.post("/upload_csv/{chat_id}")
async def upload_csv_to_chat(chat_id: str, request: CSVUploadRequest, request_obj: Request):
    """Upload CSV file to specific chat session"""
    if not state.agent or not state.executor:
        return JSONResponse(
            status_code=500,
            content={"error": "Agent or executor not initialized"}
        )
    
    try:
        # Get user from session
        user = get_current_user(request_obj)
        
        # Get user and chat specific directory
        user_path = get_user_path(user)
        chat_dir = user_path / chat_id
        
        # Verify chat exists and belongs to user
        if not chat_dir.exists():
            return JSONResponse(
                status_code=404,
                content={"error": f"Chat {chat_id} not found for user {user}"}
            )
        
        # Update state output directory
        state.output_dir = chat_dir
        
        # Copy file to chat directory
        file_name = Path(request.file_path).name
        destination = chat_dir / file_name
        shutil.copy2(request.file_path, destination)
        
        # Create query for agent to read the file with chat-specific DataFrame name
        load_query = f"read the CSV file named '{destination}' and load it into a DataFrame named 'df_{chat_id}'"
        
        # Process query using agent
        result = await Console(
            state.agent.run_stream(task=load_query)
        )
        
        return JSONResponse(
            content={
                "message": "CSV file uploaded and loaded",
                "file_name": file_name,
                "chat_id": chat_id,
                "chat_dir": str(chat_dir),
                "user": user,
                "agent_response": result.messages[-1].content if result.messages else ""
            },
            headers={
                "Access-Control-Allow-Origin": "http://localhost:3000/",
                "Access-Control-Allow-Credentials": "true",
            }
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Failed to process CSV file: {str(e)}",
                "chat_id": chat_id
            },
            headers={
                "Access-Control-Allow-Origin": "http://localhost:3000/",
                "Access-Control-Allow-Credentials": "true",
            }
        )

    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8018)

    #uvicorn main:app --reload --port 8012 --host 127.0.0.1

# curl -X POST http://localhost:8012/query \
# -H "Content-Type: application/json" \
# -d '{"query": "create a simple plot showing a sine wave"}'

#curl -X POST http://localhost:8017/restart
# curl -X GET http://localhost:8017/variables