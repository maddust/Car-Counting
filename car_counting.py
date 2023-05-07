import cv2
import json
from ultralytics import YOLO
import supervision as sv
import sys



def load_config(file_path):
    with open(file_path, 'r') as f:
        config = json.load(f)
    return config

def main():
    config = load_config("config2.json")

    # Read video source from config
    VIDEO_SOURCE = config["video_source"]

    # car, motocycle, bus, truck
    CLASS_ID = [2, 3, 5, 7]

    # Load Model
    model = YOLO("yolov8s.pt")

    # Get video size to export to video file
    video_info = sv.VideoInfo.from_video_path(VIDEO_SOURCE)
    video_size = (video_info.width, video_info.height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps=video_info.fps
    print(type(fps))
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (video_size), True)

    # Init trigger lines
    line_zones = []
    for l_zone in config["line_zones"]:
        start = sv.Point(*l_zone["start"])
        end = sv.Point(*l_zone["end"])
        line_zone = sv.LineZone(start=start, end=end)
        line_zones.append(line_zone)

    # Init annotator to draw trigger lines
    line_zone_annotator = sv.LineZoneAnnotator( thickness=4, 
                                                color=sv.Color.from_hex(color_hex="#00ff00"),
                                                text_thickness=2, 
                                                text_scale=0.6,
                                                text_offset=1.0,
                                                text_padding=3)

    # Init box to draw objects
    box_annotator = sv.BoxAnnotator(
                                    thickness=2,
                                    text_thickness=1,
                                    text_scale=0.5,
                                    text_padding=3)

    for result in model.track(VIDEO_SOURCE, show=False, stream=True, classes=CLASS_ID):
        # Get frame
        frame = result.orig_img
        detections = sv.Detections.from_yolov8(result)
        CLASS_NAMES_DICT = model.model.names

        # Set box ID so that we can count object when it cross trigger lines
        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

        labels = []
        for i, detection in enumerate(detections):
            tracker_id = detection[4] if detection[4] is not None else ""
            class_id = int((detection[3]))
            confidence = round(float(detection[2]) * 100)
            class_names = CLASS_NAMES_DICT.get(class_id)
            label_format = f"{tracker_id} {class_names}"
            labels.append(label_format)

        # Draw box around objects
        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

        # Set triggers and draw trigger lines
        for line_zone in line_zones:
            line_zone.trigger(detections=detections)
            line_zone_annotator.annotate(frame=frame, line_counter=line_zone)

        # Write to file
        out.write(frame)

        # Show results as processing
        cv2.imshow("result", frame)

        if(cv2.waitKey(1) == 27):
            break

if __name__ == "__main__":
    main()
