import os
from moviepy import VideoFileClip

home = os.path.expanduser("~")
input_path = os.path.join(home, "Desktop/new_lab/Dr_Brendan/Abdullah.avi")
output_path = os.path.join(home, "Desktop/new_lab/Dr_Brendan/Abdullah.mp4")

def convert_avi_to_mp4(input_f, output_f):
    if not os.path.exists(input_f):
        print("Dosya yok")
        return

    video = VideoFileClip(input_f)
    video.write_videofile(output_f, codec="libx264")
    video.close()

convert_avi_to_mp4(input_path, output_path)
