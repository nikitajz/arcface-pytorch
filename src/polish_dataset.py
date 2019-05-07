"""
Polish dataset faces by Mtcnn Detector
"""

import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import environments
from data.dataset import get_dataset
from face_detection import MtcnnDetector
from face_detection.mtcnn_detector import mx
from utils.logger import get_logger

logger = get_logger(__name__)


class NotDetectionError(Exception):
    pass


def clip_most_humanise_image(detector, image, target_size, padding_ratio):
    """
    画像の中で最も人間である確率の高い画像を一枚だけ返す.
    :param MtcnnDetector detector:
    :param image:
    :param int target_size:
    :param float padding_ratio:
    :return:
    """
    result = detector.detect_face(image)
    if result is None:
        raise NotDetectionError

    boxes, points = result
    chips = detector.extract_image_chips(image, points, desired_size=target_size, padding=padding_ratio)
    face_scores = boxes[:, -1]
    face_center = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
    box_areas = [(b[0] - b[2]) * (b[1] - b[3]) for b in boxes]
    max_idx = np.argmax(box_areas)
    best_face = chips[max_idx]
    best_score = face_scores[max_idx]
    return best_face, best_score


def get_arguments():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            description=__doc__)
    parser.add_argument('-d', '--dataset', required=True, type=str, help='polish target dataset name')
    parser.add_argument('--gpu', action='store_true', help='If add, use gpu to clipping image')
    parser.add_argument('--size', type=int, default=environments.INPUT_SHAPE[1], help='clipped image size')
    parser.add_argument('--padding', type=float, default=.3, help='padding ratio between face and clipped image')
    return vars(parser.parse_args())


def main():
    args = get_arguments()
    logger.info(args)
    dataset = get_dataset(args.get('dataset', None))
    print(dataset.root_path)

    img_paths = dataset.df_meta[dataset.img_colname].to_list()
    if dataset.relative_path:
        img_paths = [os.path.join(dataset.root_path, p) for p in img_paths]

    new_dataset_name = f'{dataset.name()}_polished'
    root_path = os.path.join(environments.DATASET_DIR, new_dataset_name)
    os.makedirs(root_path, exist_ok=True)

    target_size = args.get('size')

    if args.get('gpu', False):
        logger.info('use gpu')
        ctx = mx.gpu()
    else:
        ctx = mx.cpu()
    detector = MtcnnDetector(minsize=100, num_worker=1, ctx=ctx)

    results = []
    for p in tqdm(img_paths, total=len(img_paths)):
        img = cv2.imread(p)
        dirname = p.split('/')[-2]
        filename = p.split('/')[-1].split('.')[0]
        dir_path = os.path.join(root_path, dirname)
        os.makedirs(dir_path, exist_ok=True)

        try:
            clipped, prob = clip_most_humanise_image(detector, img, target_size=target_size, padding_ratio=.3)
            new_path = os.path.join(dir_path, f'{filename}x{target_size}_{prob:.3f}.jpg')
            cv2.imwrite(new_path, clipped)
            results.append([os.path.relpath(new_path, root_path), prob])
        except NotDetectionError as e:
            print(e)
            results.append([None, -1])

    df_meta = pd.DataFrame(results, columns=['img_path', 'prob'])
    df_meta['origin_path'] = dataset.df_meta[dataset.img_colname]
    df_meta[dataset.label_colname] = dataset.df_meta[dataset.label_colname]
    df_meta.to_csv(os.path.join(root_path, 'meta.csv'), index=False)


if __name__ == '__main__':
    main()
