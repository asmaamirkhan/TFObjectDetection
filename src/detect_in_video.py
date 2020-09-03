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

    # open video
    capture = cv.VideoCapture(args.input_video)

    if args.output_video:
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        output = cv.VideoWriter(args.output_video, fourcc,
                                20.0, (int(capture.get(3)), int(capture.get(4))))

    frame_counter = 0
    while True:
        # read frame by frame
        r, frame = capture.read()
        frame_counter += 1

        # the end of the video?
        if frame is None:
            break

        key = cv.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        # real object detection
        objects = detector.detect_objects(frame, args.threshold)

        # draw results
        for obj in objects:
            # draw rectangle
            cv.rectangle(frame,
                         (obj["x1"], obj["y1"]),
                         (obj["x2"], obj["y2"]),
                         (255, 255, 255))

            # draw score and class
            cv.putText(frame,
                       '{:.2f} {}'.format(obj["score"], obj["id"]),
                       (obj["x2"], obj["y2"]),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6,
                       (255, 255, 0), thickness=2)
        # show image
        cv.imshow('output', frame)

    # if image will be saved then save it
        if args.output_video:
            output.write(frame)

    # when any key has been pressed then close window and stop the program
    if args.output_video:
        logger.info(
            "Image has been saved successfully at {} path".format(args.output_image))


if __name__ == "__main__":
    # creating argument parser
    parser = argparse.ArgumentParser(description='Object Detection parameters')

    # adding arguments
    parser.add_argument('-i',
                        '--input_video',
                        help='Path to your video',
                        type=str,
                        required=True)
    parser.add_argument('-m',
                        '--model_path',
                        help='Path to .pb model',
                        type=str,
                        required=True)
    parser.add_argument('-o',
                        '--output_video',
                        help='Output file path',
                        type=str)
    parser.add_argument('-t',
                        '--threshold',
                        help='Object detection confidence',
                        default=0.7,
                        type=float)
    args = parser.parse_args()

    # if input image path is invalid then stop
    assert os.path.isfile(args.input_video), 'Invalid input file'

    # if output directory is invalid then stop
    if args.output_video:
        assert os.path.isdir(os.path.dirname(
            args.output_video)), 'No such directory'

    main(args)
