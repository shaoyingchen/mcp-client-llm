from mcp.server.lowlevel import Server
import mcp.types as types
import pyautogui
import pyperclip
import time
import asyncio


async def open_local_app(app_name: str) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """打开本机应用.

    Args:
        app_name: 应用名称
    """
    # 打开 Windows 搜索栏
    pyautogui.hotkey('win', 's')
    # 使用 pyautogui 点击 Alt + 空格键
    # pyautogui.hotkey('alt', 'space')
    time.sleep(1)  # 等待搜索栏出现
    # 将文本复制到剪贴板
    pyperclip.copy(app_name)
    # 模拟 Ctrl + V 快捷键进行粘贴
    pyautogui.hotkey('ctrl', 'v')
    time.sleep(1)  # 等待输入完成
    # 按下回车键
    pyautogui.press('enter')
    response_text = "Application opened."
    return [types.TextContent(type="text", text=response_text)]


async def lock_screen() -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    pyautogui.hotkey('win', 'l')
    response_text = "Screen locked."
    return [types.TextContent(type="text", text=response_text)]


app = Server('rpa')


@app.call_tool()
async def rpa_tool(
        name: str, arguments: dict
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    if name == "rpa":
        if "app_name" not in arguments:
            raise ValueError("Missing required argument 'app_name'")
        return await open_local_app(arguments["app_name"])
    elif name == "lock_screen":
        return await lock_screen()
    raise ValueError(f"Unknown tool name: {name}")


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="rpa",
            description="open local Apps",
            inputSchema={
                "type": "object",
                "required": ["app_name"],
                "properties": {
                    "app_name": {
                        "type": "string",
                        "description": "app name",
                    }
                },
            },
        ),
        types.Tool(
            name="lock_screen",
            description="lock screen",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        )
    ]


if __name__ == "__main__":
    from mcp.server.stdio import stdio_server


    async def arun():
        async with stdio_server() as streams:
            await app.run(
                streams[0], streams[1], app.create_initialization_options()
            )


    asyncio.run(arun())
