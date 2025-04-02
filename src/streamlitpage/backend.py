from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from datetime import datetime
import streamlit as st
import asyncio
import json
import logging
import os
import shutil
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

import httpx
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Configuration:
    """Manages configuration and environment variables for the MCP client."""

    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        self.load_env()
        self.api_key = os.getenv("API_KEY")
        self.api_point = os.getenv("API_POINT")
        self.model = os.getenv("MODEL")

    @staticmethod
    def load_env() -> None:
        """Load environment variables from .env file."""
        load_dotenv()

    @staticmethod
    def load_config(file_path: str) -> dict[str, Any]:
        """Load server configuration from JSON file.

        Args:
            file_path: Path to the JSON configuration file.

        Returns:
            Dict containing server configuration.

        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            JSONDecodeError: If configuration file is invalid JSON.
        """
        with open(file_path, "r") as f:
            return json.load(f)

    @property
    def llm_api_key(self) -> str:
        """Get the LLM API key.

        Returns:
            The API key as a string.

        Raises:
            ValueError: If the API key is not found in environment variables.
        """
        if not self.api_key:
            raise ValueError("LLM_API_KEY not found in environment variables")
        return self.api_key

    @property
    def llm_api_point(self) -> str:
        """Get the LLM API point.

        Returns:
            The API point as a string.

        Raises:
            ValueError: If the API point is not found in environment variables.
        """
        if not self.api_point:
            raise ValueError(
                "LLM_API_POINT not found in environment variables")
        return self.api_point

    @property
    def llm_model(self) -> str:
        """Get the LLM MODEL point.

        Returns:
            The API MODEL as a string.

        Raises:
            ValueError: If the API MODEL is not found in environment variables.
        """
        if not self.model:
            raise ValueError("LLM_MODEL not found in environment variables")
        return self.model


class Server:
    """Manages MCP server connections and tool execution."""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name: str = name
        self.config: dict[str, Any] = config
        self.stdio_context: Any | None = None
        self.session: ClientSession | None = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.exit_stack: AsyncExitStack = AsyncExitStack()

    async def initialize(self) -> None:
        """Initialize the server connection."""
        command = (
            shutil.which("npx")
            if self.config["command"] == "npx"
            else self.config["command"]
        )
        if command is None:
            raise ValueError(
                "The command must be a valid string and cannot be None.")

        server_params = StdioServerParameters(
            command=command,
            args=self.config["args"],
            env={**os.environ, **self.config["env"]}
            if self.config.get("env")
            else None,
        )
        try:
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.session = session
        except Exception as e:
            logging.error(f"Error initializing server {self.name}: {e}")
            await self.cleanup()
            raise

    async def list_tools(self) -> list[Any]:
        """List available tools from the server.

        Returns:
            A list of available tools.

        Raises:
            RuntimeError: If the server is not initialized.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        tools_response = await self.session.list_tools()
        tools = []

        for item in tools_response:
            if isinstance(item, tuple) and item[0] == "tools":
                for tool in item[1]:
                    tools.append(
                        Tool(tool.name, tool.description, tool.inputSchema))

        return tools

    async def execute_tool(
            self,
            tool_name: str,
            arguments: dict[str, Any],
            retries: int = 2,
            delay: float = 1.0,
    ) -> Any:
        """Execute a tool with retry mechanism.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
            retries: Number of retry attempts.
            delay: Delay between retries in seconds.

        Returns:
            Tool execution result.

        Raises:
            RuntimeError: If server is not initialized.
            Exception: If tool execution fails after all retries.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        attempt = 0
        while attempt < retries:
            try:
                logging.info(f"Starting execution of tool: {tool_name}")
                logging.info(f"Tool arguments: {arguments}")
                logging.info(f"Current attempt: {attempt + 1}/{retries}")

                # 记录调用前的状态
                logging.info(
                    f"Server {self.name} session state: {self.session is not None}")

                result = await self.session.call_tool(tool_name, arguments)
                logging.info(f"Tool execution completed successfully")
                return result

            except Exception as e:
                attempt += 1
                logging.error(f"Error executing tool: {str(e)}")
                logging.error(f"Error type: {type(e).__name__}")
                logging.error(f"Error details: {e}")
                logging.warning(
                    f"Attempt {attempt} of {retries} failed. "
                    f"Waiting {delay} seconds before retry..."
                )
                if attempt < retries:
                    await asyncio.sleep(delay)
                else:
                    logging.error("Max retries reached. Failing.")
                    raise

    async def cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            try:
                await self.exit_stack.aclose()
                self.session = None
                self.stdio_context = None
            except Exception as e:
                logging.error(
                    f"Error during cleanup of server {self.name}: {e}")


class Tool:
    """Represents a tool with its properties and formatting."""

    def __init__(
            self, name: str, description: str, input_schema: dict[str, Any]
    ) -> None:
        self.name: str = name
        self.description: str = description
        self.input_schema: dict[str, Any] = input_schema

    def format_for_llm(self) -> str:
        """Format tool information for LLM.

        Returns:
            A formatted string describing the tool.
        """
        args_desc = []
        if "properties" in self.input_schema:
            for param_name, param_info in self.input_schema["properties"].items():
                arg_desc = (
                    f"- {param_name}: {param_info.get('description', 'No description')}"
                )
                if param_name in self.input_schema.get("required", []):
                    arg_desc += " (required)"
                args_desc.append(arg_desc)

        return f"""
Tool: {self.name}
Description: {self.description}
Arguments:
{chr(10).join(args_desc)}
"""


class LLMClient:
    """Manages communication with the LLM provider."""

    def __init__(self, api_point: str, api_key: str, model: str) -> None:
        self.api_key: str = api_key
        self.api_point: str = api_point
        self.model: str = model

    def get_response(self, messages: list[dict[str, str]]) -> str:
        """Get a response from the LLM.

        Args:
            messages: A list of message dictionaries.

        Returns:
            The LLM's response as a string.

        Raises:
            httpx.RequestError: If the request to the LLM fails.
        """
        # API端点和密钥
        url = self.api_point

        # 定义请求头部
        headers = {
            'Content-Type': 'application/json',
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "messages": messages,
            "model": self.model,
            "temperature": 0.7,
            "max_tokens": 4096,
            "top_p": 1,
            "stream": False,
            "stop": None,
        }

        try:
            # 设置更长的超时时间，因为本地模型推理可能需要更长时间
            timeout_settings = httpx.Timeout(120.0, connect=60.0)
            with httpx.Client(timeout=timeout_settings) as client:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                logging.info(f"LLM response: {data}")
                return data["choices"][0]["message"]["content"]

        except httpx.RequestError as e:
            error_message = f"Error getting LLM response: {str(e)}"
            logging.error(error_message)

            if isinstance(e, httpx.HTTPStatusError):
                status_code = e.response.status_code
                logging.error(f"Status code: {status_code}")
                logging.error(f"Response details: {e.response.text}")

            return (
                f"I encountered an error: {error_message}. "
                "Please try again or rephrase your request."
            )


class ChatSession:
    """Orchestrates the interaction between user, LLM, and tools."""

    def __init__(self, servers: list[Server], llm_client: LLMClient) -> None:
        self.servers: list[Server] = servers
        self.llm_client: LLMClient = llm_client
        # 初始化session state来存储消息历史
        if "messages" not in st.session_state:
            st.session_state.messages = []

    async def cleanup_servers(self) -> None:
        """Clean up all servers properly."""
        cleanup_tasks = []
        for server in self.servers:
            cleanup_tasks.append(asyncio.create_task(server.cleanup()))

        if cleanup_tasks:
            try:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            except Exception as e:
                logging.warning(f"Warning during final cleanup: {e}")

    async def process_llm_response(self, llm_response: str) -> str:
        """Process the LLM response and execute tools if needed.

        Args:
            llm_response: The response from the LLM.

        Returns:
            The result of tool execution or the original response.
        """
        import json

        try:
            tool_call = json.loads(llm_response)
            if "tool" in tool_call and "arguments" in tool_call:
                logging.info(f"Executing tool: {tool_call['tool']}")
                logging.info(f"With arguments: {tool_call['arguments']}")

                for server in self.servers:
                    tools = await server.list_tools()
                    if any(tool.name == tool_call["tool"] for tool in tools):
                        try:
                            result = await server.execute_tool(
                                tool_call["tool"], tool_call["arguments"]
                            )

                            if isinstance(result, dict) and "progress" in result:
                                progress = result["progress"]
                                total = result["total"]
                                percentage = (progress / total) * 100
                                logging.info(
                                    f"Progress: {progress}/{total} "
                                    f"({percentage:.1f}%)"
                                )

                            return f"Tool execution result: {result}"
                        except Exception as e:
                            error_msg = f"Error executing tool: {str(e)}"
                            logging.error(error_msg)
                            return error_msg

                return f"No server found with tool: {tool_call['tool']}"
            return llm_response
        except json.JSONDecodeError:
            return llm_response

    async def start(self) -> dict[str, Any]:
        """Main chat session handler."""
        for server in self.servers:
            try:
                await server.initialize()
            except Exception as e:
                logging.error(f"Failed to initialize server: {e}")
                await self.cleanup_servers()
                return

        all_tools = []
        for server in self.servers:
            tools = await server.list_tools()
            all_tools.extend(tools)

        tools_description = "\n".join(
            [tool.format_for_llm() for tool in all_tools])

        system_message = (
            "You are a helpful assistant with access to these tools:\n\n"
            f"{tools_description}\n"
            "Choose the appropriate tool based on the user's question. "
            "If no tool is needed, reply directly.\n\n"
            "IMPORTANT: When you need to use a tool, you must ONLY respond with "
            "the exact JSON object format below, nothing else:\n"
            "{\n"
            '    "tool": "tool-name",\n'
            '    "arguments": {\n'
            '        "argument-name": "value"\n'
            "    }\n"
            "}\n\n"
            "After receiving a tool's response:\n"
            "1. Transform the raw data into a natural, conversational response\n"
            "2. Keep responses concise but informative\n"
            "3. Focus on the most relevant information\n"
            "4. Use appropriate context from the user's question\n"
            "5. Avoid simply repeating the raw data\n\n"
            "Please use only the tools that are explicitly defined above."
        )
        messages = [{"role": "system", "content": system_message}]

        return messages


class Session:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.chat_session: ChatSession | None = None


session = Session()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时执行
    config = Configuration()
    server_config = config.load_config("servers_config.json")
    servers = [
        Server(name, srv_config)
        for name, srv_config in server_config["mcpServers"].items()
    ]
    llm_client = LLMClient(config.llm_api_point,
                           config.llm_api_key, config.llm_model)
    chat_session = ChatSession(servers, llm_client)
    messages = await chat_session.start()

    session.messages = messages
    session.chat_session = chat_session
    yield
    # 关闭时执行
    await chat_session.cleanup_servers()

app = FastAPI(lifespan=lifespan)


class Message(BaseModel):
    role: str
    content: str
    timestamp: Optional[str] = None


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    timestamp: str


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        logging.info("\nUser: %s", session.messages)
        session.messages.append(
            {"role": "user", "content": request.message})
        chatSession = session.chat_session
        # 这里可以添加实际的AI处理逻辑
        # 目前返回模拟响应
        llm_response = chatSession.llm_client.get_response(session.messages)
        logging.info("\nAssistant: %s", llm_response)

        result = await chatSession.process_llm_response(llm_response)

        response = llm_response
        if result != llm_response:
            session.messages.append(
                {"role": "assistant", "content": llm_response})
            session.messages.append({"role": "system", "content": result})

            final_response = chatSession.llm_client.get_response(
                session.messages)
            logging.info("\nFinal response: %s", final_response)
            session.messages.append(
                {"role": "assistant", "content": final_response}
            )
            response = final_response
        else:
            session.messages.append(
                {"role": "assistant", "content": llm_response})

        return ChatResponse(
            response=response,
            timestamp=datetime.now().strftime("%H:%M:%S")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
