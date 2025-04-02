import time
from mcp.server.fastmcp import FastMCP
import pyautogui
import pyperclip

mcp = FastMCP()


@mcp.tool()
def call_local_app(app_name: str):
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
