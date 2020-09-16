# author: Asmaa Mirkhan ~ 2020

import os
import argparse
import cv2 as cv
from DetectorAPI import Detector
from tf_logger import logger


def main(args):
    # assign model path and threshold
    model_path = args.model_path
    threshold = args.threshold

    # create detection object
    detector = Detector(model_path=model_path, name="detection")

    # open image
    image = cv.imread(args.input_image)

    # real object detection
    objects = detector.detect_objects(image, args.threshold)

    # draw results
    for obj in objects:
        # draw rectangle
        cv.rectangle(image,
                     (obj["x1"], obj["y1"]),
                     (obj["x2"], obj["y2"]),
                     (255, 255, 255))

        # draw score and class
        cv.putText(image,
                   '{:.2f} {}'.format(obj["score"], obj["id"]),
                   (obj["x2"], obj["y2"]),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6,
                   (255, 255, 0), thickness=2)

    # show image
    cv.imshow('output', image)

    # if image will be saved then save it
    if args.output_image:
        cv.imwrite(args.output_image, image)
        logger.info(
            "Image has been saved successfully at {} path".format(args.output_image))

    # when any key has been pressed then close window and stop the program
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    # creating argument parser
    parser = argparse.ArgumentParser(description='Object detection parameters')

    # adding arguments
    parser.add_argument('-i',
                        '--input_image',
                        help='Path to your image',
                        type=str,
                        required=True)
    parser.add_argument('-m',
                        '--model_path',
                        help='Path to .pb model',
                        type=str,
                        required=True)
    parser.add_argument('-o',
                        '--output_image',
                        help='Output file path',
                        type=str)
    parser.add_argument('-t',
                        '--threshold',
                        help='Object detection confidence',
                        default=0.7,
                        type=float)
    args = parser.parse_args()

    # if input image path is invalid then stop
    assert os.path.isfile(args.input_image), 'Invalid input file'

    # if output directory is invalid then stop
    if args.output_image:
        assert os.path.isdir(os.path.dirname(
            args.output_image)), 'No such directory'

    main(args)
