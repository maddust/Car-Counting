# Car-Counting

To run

``python main.py path-to-video``

This project is pretty simple thanks to yolov8 with new implementation of tracker.
You can change tracking algorithm with

* BoT-SORT - botsort.yaml

* ByteTrack - bytetrack.yaml

``results = model.track(source="https://youtu.be/Zgi9g1ksQHc", tracker="bytetrack.yaml")``

If you have GPU, 

## How it works
1. Set trigger position
2. Load the model (of course)
3. Iterate frame by frame of the video
    1. Set unique ID for each object
    2. Check if any object cross the trigger
    3. Draw result
4. End

[YOUTUBE VIDEO](https://youtu.be/bf5nmbEaXhU)

Note: this project uses [supervision](https://github.com/roboflow/supervision) to detect when objects cross the trigger.
It's on beta and many thing has changed so their documentation is outdated at the moment. You should watch their latest video instead.

The video used to test is recorded from [camstreamer](https://camstreamer.com/live/stream/27159-4-corners-camera-downtown)
