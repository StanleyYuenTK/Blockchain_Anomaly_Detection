import inspect
import sys
sys.path.append("..")
import gnn_zoo  # 假設你的檔名是 final_ki.py

functions_list = inspect.getmembers(gnn_zoo, inspect.isfunction)
for name, _ in functions_list:
    print(name)