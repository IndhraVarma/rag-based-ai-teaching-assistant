#Converts video to mp3
import os
import subprocess

files=os.listdir("Videos")
print(files)
for file in files:
    name= os.path.splitext(file)[0]
    number, title= name.split(". ",1);

    print(number, " ", title)
    subprocess.run(["ffmpeg", "-i", f"Videos/{file}", f"audios/{number}_{title}.mp3"])