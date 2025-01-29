import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
from typing import Dict, Optional
import json
import uuid
import uvicorn
import logging
from contextlib import asynccontextmanager
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.code_execution import PythonCodeExecutionTool
from autogen_ext.code_executors.jupyter import JupyterCodeExecutor
from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock
import socket
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global connection tracking
active_connections: Dict[str, WebSocket] = {}
client_to_session: Dict[str, str] = {}

class ChatSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.jupyter_executor = None
        self.agent = None
        self.output_dir = Path(f"chat_outputs/{session_id}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.messages = []
        self._initialized = False
        
    async def initialize(self):
        try:
            # Initialize Jupyter executor
            self.jupyter_executor = JupyterCodeExecutor(
                kernel_name="python3",
                timeout=120,
                output_dir=self.output_dir
            )
            await self.jupyter_executor.__aenter__()
            
            # Initialize agent
            tool = PythonCodeExecutionTool(self.jupyter_executor)
            self.agent = AssistantAgent(
                "assistant",
                OpenAIChatCompletionClient(model="gpt-4-mini"),
                tools=[tool],
                reflect_on_tool_use=True,
                system_message="""
                You are a data analysis assistant. Maintain state between queries by:
                1. First check if required variables exist before recreating them
                2. Reference previously created variables and dataframes when relevant
                3. Let the user know what data is currently available in the session
                4. When creating new analysis, build upon existing data when possible
                
                Before every import, include a code to install dependency use the following code example:
                import subprocess
                subprocess.check_call(['pip', 'install', 'package_name'])
                """
            )
            self._initialized = True
            logger.info(f"Chat session {self.session_id} initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize chat session {self.session_id}: {str(e)}")
            raise

    async def cleanup(self):
        if self.jupyter_executor and self._initialized:
            try:
                await self.jupyter_executor.__aexit__(None, None, None)
                self._initialized = False
                logger.info(f"Chat session {self.session_id} cleaned up successfully")
            except Exception as e:
                logger.error(f"Error during cleanup of session {self.session_id}: {str(e)}")
    
    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        logger.debug(f"Added {role} message to session {self.session_id}")

# Store active sessions
sessions: Dict[str, ChatSession] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up FastAPI application...")
    yield
    # Shutdown
    logger.info("Shutting down FastAPI application...")
    for session in sessions.values():
        await session.cleanup()

# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)

# Create necessary directories
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
Path("chat_outputs").mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

async def get_variables(session: ChatSession) -> list:
    """Get available variables in the session"""
    if session.jupyter_executor:
        try:
            cancellation_token = CancellationToken()
            result = await session.jupyter_executor.execute_code_blocks(
                [CodeBlock(code="list(locals().keys())", language="python")],
                cancellation_token
            )
            return result.output if result.output else []
        except Exception as e:
            logger.error(f"Error getting variables for session {session.session_id}: {str(e)}")
            return []
    return []

async def broadcast_files(websocket: WebSocket, session: ChatSession):
    """Send file list to client"""
    try:
        files = [f.name for f in session.output_dir.glob("*") if f.is_file()]
        await websocket.send_json({
            "type": "files",
            "files": files
        })
    except Exception as e:
        logger.error(f"Error broadcasting files for session {session.session_id}: {str(e)}")
# Basic routes
@app.get("/")
async def get_root():
    return FileResponse('static/index.html')

@app.get("/health")
async def health_check():
    return JSONResponse({"status": "healthy"})

# WebSocket endpoint
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    try:
        await websocket.accept()
        active_connections[client_id] = websocket
        logger.info(f"WebSocket connection accepted for client: {client_id}")
        
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connection_status",
            "status": "connected",
            "client_id": client_id
        })
        
        while True:
            try:
                data = await websocket.receive_json()
                logger.info(f"Received message from {client_id}: {data}")
                
                if data["type"] == "new_chat":
                    # Create new chat session
                    session_id = str(uuid.uuid4())
                    session = ChatSession(session_id)
                    await session.initialize()
                    sessions[session_id] = session
                    client_to_session[client_id] = session_id
                    
                    await websocket.send_json({
                        "type": "new_chat",
                        "chat_id": session_id
                    })
                    logger.info(f"Created new chat session: {session_id}")
                    
                elif data["type"] == "message":
                    # Validate session exists
                    if client_id not in client_to_session:
                        await websocket.send_json({
                            "type": "error",
                            "error": "No active chat session"
                        })
                        continue
                        
                    session_id = client_to_session[client_id]
                    session = sessions.get(session_id)
                    if not session:
                        await websocket.send_json({
                            "type": "error",
                            "error": "Session not found"
                        })
                        continue
                    
                    # Process message
                    user_message = data.get("content", "")
                    session.add_message("user", user_message)
                    
                    try:
                        # Get agent response
                        result = await session.agent.run_stream(task=user_message)
                        response = result.messages[-1].content if result.messages else "No response"
                        
                        session.add_message("assistant", response)
                        await websocket.send_json({
                            "type": "message",
                            "content": response
                        })
                        
                        # Update client state
                        variables = await get_variables(session)
                        await websocket.send_json({
                            "type": "variables",
                            "variables": variables
                        })
                        
                        await broadcast_files(websocket, session)
                        
                    except Exception as e:
                        logger.error(f"Error processing message in session {session_id}: {str(e)}")
                        await websocket.send_json({
                            "type": "error",
                            "error": "Error processing message"
                        })
                    
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON from client {client_id}: {e}")
                await websocket.send_json({
                    "type": "error",
                    "error": "Invalid message format"
                })
                continue
                
    except WebSocketDisconnect:
        logger.info(f"Client disconnected normally: {client_id}")
    except Exception as e:
        logger.error(f"Error in websocket connection: {str(e)}")
    finally:
        # Cleanup on disconnect
        if client_id in active_connections:
            del active_connections[client_id]
        if client_id in client_to_session:
            session_id = client_to_session[client_id]
            if session_id in sessions:
                await sessions[session_id].cleanup()
                del sessions[session_id]
            del client_to_session[client_id]
        logger.info(f"Cleaned up connection for client: {client_id}")

def find_available_port(start_port: int, max_tries: int = 100) -> Optional[int]:
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_tries):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    return None

def run_server(start_port: int = 8000):
    """Run the FastAPI server"""
    port = find_available_port(start_port)
    if port is None:
        logger.error(f"Could not find an available port after trying {start_port} through {start_port + 99}")
        return
    
    logger.info(f"Starting server at http://localhost:{port}")
    try:
        uvicorn.run(
            app, 
            host="127.0.0.1", 
            port=port,
            ws_ping_interval=None,
            ws_ping_timeout=None,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"Error starting server: {e}")

if __name__ == "__main__":
    run_server()