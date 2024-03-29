# 👩‍🔬 TensorFlow Object Detection
Core code of object detection in Tensorflow, can be used to test models.

## 🙌 Available Codes
1. [detect_in_image](./src/detect_in_image.py): Detects or objects in a given image due to given Tensorflow model
2. [detect_in_video](./src/detect_in_video.py): Detects and objects in a given video due to given Tensorflow model

> 👮‍♀️ Make sure that you have OpenCV and Tensorflow already installed

## 👩‍🚀 Usage 
1. Clone or download this repo
2. Open [src](/src) folder in CMD

### For `detect_in_image.py`:
3. Run:
```bash
   python detect_in_image.py --input_image <PATH_TO_INPUT_JPG_FILE> --output_image <PATH_TO_OUTPUT_JPG_FILE>  --model_path <PATH_TO_INPUT_PB_FILE> --threshold <THRESHOLD>
```

### For `detect_in_video.py`:
3. Run:
```bash
python detect_in_video.py --input_video <PATH_TO_INPUT_MP4_FILE> --output_image <PATH_TO_OUTPUT_MP4_FILE>  --model_path <PATH_TO_INPUT_PB_FILE> --threshold <THRESHOLD>
```

4. TADAA 🎉 It's done 🤗
5. Press <kbd>Q</kbd> to stop

> To see running options run _for all codes_:
>   `python detect_in_video.py --help`

👮‍♀️ Thrshold value should be between 0 and 1.

## 🏗️ Used by
- [👧 BlurryFaces](https://github.com/asmaamirkhan/BlurryFaces)
- [🐾 ObjectTrackers](https://github.com/asmaamirkhan/ObjectTracker-s-)

## 💼 Contact
Find me on [LinkedIn](https://www.linkedin.com/in/asmaamirkhan/) and feel free to mail me, [Asmaa 🦋](mailto:asmaamirkhan.am@gmail.com)
