import inspect
import sys
sys.path.append("..")
import GNNs  # 假設你的檔名是 final_ki.py

functions_list = inspect.getmembers(GNNs, inspect.isfunction)
for name, _ in functions_list:
    print(name)