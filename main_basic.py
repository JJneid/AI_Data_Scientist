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

async def save_files(file_name: str, content: str, file_type: str = 'txt') -> str:
    """
    Save content to a file with proper formatting and naming convention.
    
    Args:
        file_name (str): Base name for the file (will be cleaned and formatted)
        content (str): Content to save
        file_type (str): File extension (default: 'txt')
        directory (Path): Directory to save file (default: None, uses state.output_dir)
    
    Returns:
        str: Path to saved file
    """
    try:
        # Use state output directory if none provided
        save_dir = Path(path)
        
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
                # If content is already a DataFrame
                content.to_csv(file_path, index=False)
            elif isinstance(content, str):
                # If content is a CSV string
                with open(file_path, 'w') as f:
                    f.write(content)
            elif isinstance(content, (list, dict)):
                # If content is a list or dict, convert to DataFrame
                df = pd.DataFrame(content)
                df.to_csv(file_path, index=False)
            else:
                raise ValueError(f"Unsupported CSV content type: {type(content)}")
                
        elif file_type in ['png', 'jpg', 'jpeg']:
            # For binary files like images
            with open(file_path, 'wb') as f:
                f.write(content if isinstance(content, bytes) else content.encode())
        
        else:
            # For text files
            with open(file_path, 'w') as f:
                f.write(str(content))
        
        print(f"File saved successfully at: {file_path}")
        return str(file_path)
    
    except Exception as e:
        print(f"Error saving file: {str(e)}")
        raise


save_tool = FunctionTool(
    save_files, 
    description="save to a file",
    name="save_tool"  # Explicitly set the name
)



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
            When asked to save and download files, use the tool `save_tool`


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
        "http://localhost:8010",
        "http://127.0.0.1:8010",
        "http://[::1]:8010",  # IPv6 localhost
        "http://0.0.0.0:8010",
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
                "Access-Control-Allow-Origin": "http://localhost:8010",
                "Access-Control-Allow-Credentials": "true",
            }
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
            headers={
                "Access-Control-Allow-Origin": "http://localhost:8010",
                "Access-Control-Allow-Credentials": "true",
            }
        )
@app.get("/variables")
async def get_variables():
    """Get list of available variables in the kernel"""
    if not state.executor:
        return {"variables": []}
    
    try:
        cancellation_token = CancellationToken()
        result = await state.executor.execute_code_blocks(
            [CodeBlock(
                code="print([var for var in locals().keys() if not var.startswith('_')])", 
                language="python"
            )],
            cancellation_token
        )
        
        # Clean up the output and convert to list
        variables = result.output
        if isinstance(variables, str):
            # Remove any leading/trailing brackets and split
            variables = variables.strip('[]').replace("'", "").split(', ')
            variables = [v.strip() for v in variables if v.strip()]
        
        return {"variables": variables}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

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

from fastapi.responses import FileResponse
from fastapi import HTTPException

@app.get("/file/{file_path:path}")
async def get_file_content(file_path: str):
    """Get content of a specific file"""
    full_path = state.output_dir / file_path
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        # If it's an image file, return it directly
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            return FileResponse(full_path)
        
        # For other files, return the content as before
        content = full_path.read_text()
        return {"content": content}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading file: {str(e)}"
        )


import shutil  # Add this to your imports at the top

@app.post("/restart")
async def restart_kernel():
    """Restart the Jupyter kernel and clear the coding directory"""
    if not state.executor:
        return JSONResponse(
            status_code=500,
            content={"error": "Executor not initialized"}
        )
    
    try:
        # First, clear all files in the coding directory
        if state.output_dir.exists():
            # Remove all contents of the directory
            for item in state.output_dir.iterdir():
                if item.is_file():
                    item.unlink()  # Delete file
                elif item.is_dir():
                    shutil.rmtree(item)  # Delete directory and its contents
            
        # Create the directory if it doesn't exist
        state.output_dir.mkdir(exist_ok=True)
        
        # Then restart the kernel
        await state.executor.restart()
        
        return JSONResponse(
            content={
                "message": "Kernel successfully restarted and coding directory cleared",
                "deleted_path": str(state.output_dir)
            },
            headers={
                "Access-Control-Allow-Origin": "http://localhost:8009",
                "Access-Control-Allow-Credentials": "true",
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to restart kernel or clear directory: {str(e)}"},
            headers={
                "Access-Control-Allow-Origin": "http://localhost:8009",
                "Access-Control-Allow-Credentials": "true",
            }
        )


class CSVUploadRequest(BaseModel):
    file_path: str

@app.post("/upload_csv")
async def upload_csv(request: CSVUploadRequest):
    """
    Upload CSV file and have the agent load it into memory
    """
    try:
        # Copy file to working directory
        file_name = Path(request.file_path).name
        destination = state.output_dir / file_name
        shutil.copy2(request.file_path, destination)
        
        # Create query for agent to load the file
        load_query = f"read the CSV file named '{destination}' and load it into a DataFrame named 'df'"
        
        # Process query using agent
        result = await Console(
            state.agent.run_stream(task=load_query)
        )
        
        return JSONResponse(
            content={
                "message": "CSV file uploaded and loaded",
                "file_name": file_name,
                "agent_response": result.messages[-1].content if result.messages else "",
            },
            headers={
                "Access-Control-Allow-Origin": "http://localhost:8010",
                "Access-Control-Allow-Credentials": "true",
            }
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to process CSV file: {str(e)}"},
            headers={
                "Access-Control-Allow-Origin": "http://localhost:8010",
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