"""LexiCare 应用包。

本目录（app/）是项目的主包。此前缺 __init__.py 会导致 app/ 内的脚本
（doc_qa_gradio.py / gradio_app.py / api.py）里 `from app.xxx import ...`
被误解析为 app/app.py 这个同名文件（ModuleNotFoundError: 'app' is not a package）。
"""
