


# %%
import os

folder_path = 'human'
url = "10.1.10.119:8001"

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    
    if os.path.isfile(file_path) and filename.lower().endswith(('.mp4')):
        command = f'python human_client.py -u {url} "{file_path}"'
        os.system(command)

# %%
