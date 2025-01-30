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

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.code_execution import PythonCodeExecutionTool
from autogen_ext.code_executors.jupyter import JupyterCodeExecutor
from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock

load_dotenv()

class GlobalState:
    def __init__(self):
        self.executor = None
        self.agent = None
        self.output_dir = Path("coding")

state = GlobalState()

class QueryRequest(BaseModel):
    query: str

from fastapi.middleware.cors import CORSMiddleware



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        state.executor = JupyterCodeExecutor(
            kernel_name="python3",
            timeout=120,
            output_dir=state.output_dir
        )
        await state.executor.__aenter__()
        
        tool = PythonCodeExecutionTool(state.executor)
        state.agent = AssistantAgent(
            "assistant",
            OpenAIChatCompletionClient(model="gpt-4o-mini"),
            tools=[tool],
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
            """
        )
        yield
    finally:
        # Shutdown
        if state.executor:
            await state.executor.__aexit__(None, None, None)

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8006",
        "http://127.0.0.1:8006",
        "http://[::1]:8006",  # IPv6 localhost
        "http://0.0.0.0:8006",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/query")
async def query_agent(request: QueryRequest):
    """Process a query and return the final response"""
    if not state.agent or not state.executor:
        return JSONResponse(
            status_code=500,
            content={"error": "Agent or executor not initialized"}
        )
    
    try:
        # Process the query using Console and run_stream
        result = await Console(
            state.agent.run_stream(task=request.query)
        )
        
        # Get the final response content
        response_content = result.messages[-1].content if result.messages else ""
        
        return JSONResponse(
            content={"response": response_content},
            headers={
                "Access-Control-Allow-Origin": "http://localhost:8006",
                "Access-Control-Allow-Credentials": "true",
            }
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
            headers={
                "Access-Control-Allow-Origin": "http://localhost:8006",
                "Access-Control-Allow-Credentials": "true",
            }
        )
@app.get("/variables")
async def get_variables():
    """Get list of available variables in the kernel"""
    if not state.executor:
        raise HTTPException(status_code=500, detail="Executor not initialized")
    
    try:
        cancellation_token = CancellationToken()
        result = await state.executor.execute_code_blocks(
            [CodeBlock(code="list(locals().keys())", language="python")],
            cancellation_token
        )
        return {"variables": result.output if result.output else []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/variable/{var_name}")
async def get_variable_value(var_name: str):
    """Get value of a specific variable"""
    if not state.executor:
        raise HTTPException(status_code=500, detail="Executor not initialized")
    
    try:
        cancellation_token = CancellationToken()
        result = await state.executor.execute_code_blocks(
            [CodeBlock(code=f"print({var_name})", language="python")],
            cancellation_token
        )
        return {"value": result.output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files")
async def get_files():
    """Get list of files in the output directory"""
    if not state.output_dir.exists():
        return {"files": []}
    
    files = [str(f.relative_to(state.output_dir)) for f in state.output_dir.rglob("*") if f.is_file()]
    return {"files": files}

@app.get("/file/{file_path:path}")
async def get_file_content(file_path: str):
    """Get content of a specific file"""
    full_path = state.output_dir / file_path
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        content = full_path.read_text()
        return {"content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8013)