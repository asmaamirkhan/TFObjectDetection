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

    # open image
    image = cv.imread(args.input_image)

    # real face detection
    boxes, scores, classes, num = odapi.processFrame(image)

    # draw results
    for i, box in enumerate(boxes):
        if(scores[i] > threshold):
            x1, y1, x2, y2 = box
            cv.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv.putText(image, 'class: {}, score: {:.3f}'.format(
                classes[i], scores[i]), (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)

    # show image
    cv.imshow('output', image)

    # if image will be saved then save it
    if args.output_image:
        cv.imwrite(args.output_image, image)
        print('Image has been saved successfully at', args.output_image,
              'path')
    cv.imshow('output', image)

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
                        help='Face detection confidence',
                        default=0.7,
                        type=float)
    args = parser.parse_args()
    print(args)
    # if input image path is invalid then stop
    assert os.path.isfile(args.input_image), 'Invalid input file'

    # if output directory is invalid then stop
    if args.output_image:
        assert os.path.isdir(os.path.dirname(
            args.output_image)), 'No such directory'

    main(args)
