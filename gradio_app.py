#!/usr/bin/env python3
import http.server
import socketserver
import webbrowser
import os
from pathlib import Path

PORT = 7860
DIRECTORY = Path(__file__).parent

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIRECTORY), **kwargs)

def main():
    # 确保 chat.html 存在
    html_path = DIRECTORY / "chat.html"
    if not html_path.exists():
        print("错误：找不到 chat.html 文件，请将 chat.html 放在与 gradio_app.py 相同的目录下。")
        return
    
    # 切换到目标目录
    os.chdir(DIRECTORY)
    
    # 启动服务器
    with socketserver.TCPServer(("127.0.0.1", PORT), MyHTTPRequestHandler) as httpd:
        print(f"服务器已启动: http://127.0.0.1:{PORT}/chat.html")
        webbrowser.open(f"http://127.0.0.1:{PORT}/chat.html")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n服务器已关闭")

if __name__ == "__main__":
    main()