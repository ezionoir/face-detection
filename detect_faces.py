from argparse import ArgumentParser
from face_detection import build_detector
from tqdm import tqdm
from collections import OrderedDict
import cv2
from numpy import stack
import os
import json

from utils import get_video_paths

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--input_path', help='Path to videos')
    parser.add_argument('--output_path', help='Path for output boxes')
    parser.add_argument('--model', choices=['DSFDDetector', 'RetinaNetResNet50', 'RetinaNetMobileNetV1'], default='DSFDDetector', help='Face detection model')
    parser.add_argument('--downscale', type=int, default=1, help="Downscale ratio")
    parser.add_argument('--device', choices=['cpu', 'gpu'], required=True, help='Device (CPU/GPU)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')

    options = parser.parse_args()

    if options.device == 'gpu':
        options.device = 'cuda:0'

    detector = build_detector(name=options.model, device=options.device)

    video_paths = get_video_paths(base_path=options.input_path)

    for video_path in tqdm(video_paths):
        dict = OrderedDict()

        video = cv2.VideoCapture(video_path)

        num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        num_batches = (num_frames + options.batch_size - 1) // options.batch_size

        for i in range(num_batches):
            batch = []

            while True:
                success, frame = video.read()
                if not success:
                    break
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, dsize=(frame.shape[1] // options.downscale, frame.shape[0] // options.downscale))
                batch.append(frame)

            results = detector.batched_detect(stack(batch, axis=0))
            for j, result in enumerate(results):
                if result is not None and result.shape[0] > 0 and result.shape[1] > 0:
                    dict[i * options.batch_size + j] = result.tolist()
                else:
                    dict[i * options.batch_size + j] = []

        file_name = os.path.basename(video_path)
        with open(os.path.join(options.output_path, file_name + '.json'), 'w') as f:
            json.dump(f)