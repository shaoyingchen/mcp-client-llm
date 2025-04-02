import streamlit as st
import time
from datetime import datetime
import requests
import json

# 设置页面配置
st.set_page_config(
    page_title="聊天应用",
    page_icon="💬",
    layout="wide"
)

# API配置
API_URL = "http://localhost:8000/api/chat"

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []

# 页面标题
st.title("💬 聊天应用")

# 侧边栏
with st.sidebar:
    st.title("设置")
    # API配置
    api_url = st.text_input("API地址", value=API_URL)
    st.markdown("---")
    st.markdown("### 关于")
    st.markdown("这是一个使用Streamlit开发的聊天应用，与FastAPI后端交互。")

# 显示聊天消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "timestamp" in message:
            st.caption(message["timestamp"])

# 用户输入
if prompt := st.chat_input("在这里输入消息..."):
    # 添加用户消息
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })

    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
        st.caption(datetime.now().strftime("%H:%M:%S"))

    # 调用后端API
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                # 准备请求数据
                payload = {"message": prompt}

                # 发送请求到后端
                response = requests.post(api_url, json=payload)
                response.raise_for_status()  # 检查响应状态

                # 解析响应
                result = response.json()

                # 显示AI响应
                st.markdown(result["response"])
                st.caption(result["timestamp"])

                # 添加AI响应到消息历史
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["response"],
                    "timestamp": result["timestamp"]
                })

            except requests.exceptions.RequestException as e:
                error_message = f"API调用失败: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })

# 添加清除聊天记录的按钮
if st.button("清除聊天记录"):
    st.session_state.messages = []
    st.rerun()
