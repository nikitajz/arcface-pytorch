# Face Detection 

顔の場所と目、鼻、口の特徴点、顔である確率を予測するモデル。

Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Neural Networks の実装。

## Requirements

* mxnet
* cv2

## Usage

```python
from face_detection import MtcnnDetector
from face_detection.mtcnn_detector import mx
import cv2

image = cv2.imread("path/to/image.jpg")
dtector = MtcnnDetector(minsize=10, ctx=mx.cpu(0), num_worker=4,
                                 accurate_landmark=False)
result = detector.detect_face(image)
if result is None:
    return None

boxes, points = result
chips = detector.extract_image_chips(image, points, desired_size=target_size, padding=padding)

cv2.imwrite("result1.png", chips[0])
```