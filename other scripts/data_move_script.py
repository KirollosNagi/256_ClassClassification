from pathlib import Path
import shutil
import os

source = 'C:/Users/Kirollos Nagi/PycharmProjects/tftest/ass2/training_data'
dest = 'C:/Users/Kirollos Nagi/PycharmProjects/tftest/ass2/test_data'
dpath = Path(dest)
l =[f for f in Path().iterdir() if f.is_dir()]

for dir in l:
    ndir = dpath / dir
    ndir.mkdir()
    for i in range(15):
        m = max(dir.glob('*.jpg'))
        shutil.move(os.path.join(source,m),os.path.join(dest,ndir))
   
      
