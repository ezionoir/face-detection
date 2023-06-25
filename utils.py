import os

def get_video_paths(base_path):
    video_paths = []
    for folder in os.listdir(base_path):
        for file in os.listdir(os.path.join(base_path, folder)):
            if file.endswith('.mp4'):
                video_paths.append(os.path.join(base_path, folder, file))
    return video_paths