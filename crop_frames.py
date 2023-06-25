from argparse import ArgumentParser
import os
import json
import cv2
from multiprocessing import Pool
import tqdm
from functools import partial

from utils import get_video_paths

def crop_frames(video_path, box_folder, output_folder):
    name = os.path.basename(video_path).split('.')[0]

    dict = {}
    with open(os.path.join(box_folder, name + '.json'), 'r') as f:
        dict = json.load(f)

    os.path.mkdir(os.path.join(output_folder, name))

    video = cv2.VideoCapture(video_path)
    num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)

    for i in range(num_frames):
        success, frame = video.read()
        if not success:
            continue

        faces = dict[str(i)]
        for j, face in enumerate(faces):
            if len(face) > 0:
                left = int(face[0])
                top = int(face[1])
                right = int(face[2])
                bottom = int(face[3])

                cropped = frame[top:bottom, left:right]
                cv2.imwrite(os.path.join(output_folder, name, str(i) + '_' + str(j) + '.png'), cropped)

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--videos_path', help='Path to videos')
    parser.add_argument('--boxes path', help='Path to bounding boxes')
    parser.add_argument('--output_path', help='Path to output folder')
    parser.add_argument('--workers', help='Nmber of workers')

    options = parser.parse_args()

    video_paths = get_video_paths(base_path=options.videos_path)

    with Pool(processes=options.workers) as p:
        with tqdm(total=len(video_paths)) as pbar:
            for video in p.imap_unordered(partial(crop_frames, box_folder = options.boxes_path, output_folder = options.output_path), video_paths):
                video.update()