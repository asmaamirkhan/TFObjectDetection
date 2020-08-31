# author: Asmaa Mirkhan ~ 2020

import os
import argparse
import cv2 as cv
from DetectorAPI import DetectorAPI

def main(args):
    # assign model path and threshold
    model_path = args.model_path
    threshold = args.threshold

    # create detection object
    odapi = DetectorAPI(path_to_ckpt=model_path)

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
        boxes, scores, classes, num = odapi.processFrame(frame)

        # draw results
        for i, box in enumerate(boxes):
            if(scores[i] > threshold):
                x1, y1, x2, y2 = box
                cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv.putText(frame, 'class: {}, score: {:.3f}'.format(
                    classes[i], scores[i]), (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)

        # show image
        cv.imshow('output', frame)

    # if image will be saved then save it
        if args.output_video:
            output.write(frame)

    # when any key has been pressed then close window and stop the program
    if args.output_video:
        print('Video has been saved successfully at', args.output_video, 'path')

    cv.destroyAllWindows()


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
    print(args)
    # if input image path is invalid then stop
    assert os.path.isfile(args.input_video), 'Invalid input file'

    # if output directory is invalid then stop
    if args.output_video:
        assert os.path.isdir(os.path.dirname(
            args.output_video)), 'No such directory'

    main(args)
