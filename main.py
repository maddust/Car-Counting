import cv2
from ultralytics import YOLO
import supervision as sv
import sys



def main(VIDEO_SOURCE):

    # Points to draw trigger lines
    # Because at the current there is no implementation from supervision
    # to switch counter for IN/OUT result, and the LineZoneAnnotator results
    # depends on results from LineZone then we have to swap START
    # and END points to swap IN/OUT result
    # Road from EAST
    END_E = sv.Point(1750, 800)
    START_E = sv.Point(1280, 520)

    # Road from WEST
    START_W = sv.Point(500, 1000)
    END_W = sv.Point(200, 580)

    # Road from NORTH
    END_N = sv.Point(1280, 520)
    START_N = sv.Point(200, 580)

    # Road from SOUTH
    START_S = sv.Point(1750, 800)
    END_S = sv.Point(500, 1000)


    # car, motocycle, bus, truck
    CLASS_ID = [2, 3, 5, 7]

    # Load Model
    model = YOLO("yolov8s.pt")

    # Get video size to export to video file
    video_info = sv.VideoInfo.from_video_path(VIDEO_SOURCE)
    video_size = (video_info.width, video_info.height)
    out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'MJPG'), 20, (video_size))

    # Init trigger lines
    line_zone_E = sv.LineZone(start=START_E, end=END_E)
    line_zone_W = sv.LineZone(start=START_W, end=END_W)
    line_zone_N = sv.LineZone(start=START_N, end=END_N)
    line_zone_S = sv.LineZone(start=START_S, end=END_S)

    # Init annotator to draw trigger lines
    line_zone_annotator = sv.LineZoneAnnotator(thickness=5, color=sv.Color.from_hex(color_hex="#00ff00"),  text_thickness=2, text_scale=2)

    # Init box to draw objects
    box_annotator = sv.BoxAnnotator()

    for result in model.track(VIDEO_SOURCE, show=True, stream=True, classes=CLASS_ID):
        # Get frame
        frame = result.orig_img
        detections = sv.Detections.from_yolov8(result)

        # Set box ID so that we can count object when it cross trigger lines
        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
        
        # Draw box around objects
        frame = box_annotator.annotate(scene=frame, detections=detections)
        
        # Set triggers
        line_zone_E.trigger(detections=detections)
        line_zone_W.trigger(detections=detections)
        line_zone_N.trigger(detections=detections)
        line_zone_S.trigger(detections=detections)

        # Draw trigger lines
        line_zone_annotator.annotate(frame=frame, line_counter=line_zone_E)
        line_zone_annotator.annotate(frame=frame, line_counter=line_zone_W)
        line_zone_annotator.annotate(frame=frame, line_counter=line_zone_N)
        line_zone_annotator.annotate(frame=frame, line_counter=line_zone_S)

        # Write to file
        out.write(frame)
        # Show results as processing
        cv2.imshow("result", frame)

        if(cv2.waitKey(30) == 27):
            break


if len(sys.argv) == 2:
    main(sys.argv[1])