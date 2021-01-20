# ------------------------------------------------------------------------------
# This is an implementation of Head Count in real time in python 3 using
# Deep Learning Object Detection with YOLO_v4 and
# Object Tracking to minimize the time cost due to detection( too expensive to
# be run at each and every frame).
# Note: This is just a basic implementation and has a lot of scope for
# improvement.
# ------------------------------------------------------------------------------

import argparse
import sys
import numpy as np
import os.path
import cv2
from imutils.video import FPS
from tracking.centroidtracker import CentroidTracker
from tracking.trackableobject import TrackableObject
from fill_tracker import fill_tracker_list


def run():
    # Initialize the parameters
    conf_threshold = 0.46  # Confidence threshold
    nms_threshold = 0.6  # Non-maximum suppression threshold
    inp_width = 416  # Width of network's input image 416 - default for yolo
    inp_height = 416  # Height of network's input image 416 - default for yolo
    skip_frames = 0  # No. of frames skipped for next detection
    write_output = True
    # construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser(description='Object Detection and Tracking using YOLO in OPENCV')
    parser.add_argument("-s", "--skip_frames", type=int, default=30,
                        help="# of skip frames between detections")
    parser.add_argument("-mw", "--model_weights",
                        help="path to Yolo pre-trained model weights")
    parser.add_argument("-mc", "--model_cfg",
                        help="path to Yolo pre-trained model cfg")
    parser.add_argument("-i", "--video", type=str,
                        help="path to optional input video file")
    parser.add_argument("-o", "--output", type=str,
                        help="path to optional output video file")
    parser.add_argument("-c", "--confidence", type=float, default=0.4,
                        help="minimum probability to filter weak detections")

    args = parser.parse_args()
    video_path = "videos/zzzz.mkv"
    if args.video:
        video_path = args.video

    # Load names of classes
    classes_file = "data/coco.names"
    with open(classes_file, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    # Give the configuration and weight files for the model and load the network using them.
    model_configuration = "model_data/yolov4.cfg"
    model_weights = "model_data/yolov4.weights"
    if args.model_weights and args.model_cfg:
        model_configuration = args.model_weights
        model_weights = args.model_cfg

    if args.skip_frames:
        skip_frames = args.skip_frames

    output_file = "output_video/result.avi"
    writer = None
    if args.output:
        output_file = args.output
        write_output = True

    net = cv2.dnn.readNetFromDarknet(model_configuration, model_weights)

    # if use CUDA
    # net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    # net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL)

    # Get the names of the output layers
    def get_outputs_names(net):
        # Get the names of all the layers in the network
        layers_names = net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Process inputs
    win_name = 'Deep learning object detection in OpenCV usign YOLO_v4-tiny'
    cv2.namedWindow(win_name)

    if video_path:
        # Open the video file
        if not os.path.isfile(video_path):
            print("Input video file ", video_path, " doesn't exist")
            sys.exit(1)
        cap = cv2.VideoCapture(video_path)
    # else:
    #     # Webcam input
    #     cap = cv2.VideoCapture(0)

    # instantiate our centroid tracker, then initialize a list to store
    max_disap = 12
    max_distance = 30
    ct = CentroidTracker(maxDisappeared=max_disap, maxDistance=max_distance)
    trackers = []
    trackableObjects = {}
    total_frames = 0
    total_down = 0
    total_up = 0
    x = []
    empty = []
    empty1 = []
    # start the frames per second throughput estimator
    fps = FPS().start()
    # initialize the frame dimensions (we'll set them as soon as we read
    # the first frame from the video)
    W = None
    H = None

    # loop over frames from the video stream
    while True:
        # get frame from the video
        has_frame, frame = cap.read()
        # cv.line(frame, (43, 160), (238, 170), [0, 255, 255], 10)
        # Stop the program if reached end of video
        if not has_frame:
            print("Done processing !!!")
            print("Output file is stored as ", output_file)
            break
        # frame = imutils.resize(frame, width=640)
        # If you want track another video coment next row
        # frame = frame[40:420, 1200:1480]
        # converting frame form BGR to RGB for dlib
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # if the frame dimensions are empty, set them
        if not W and not H:
            H, W = frame.shape[:2]

        # initialize writer
        if write_output and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(output_file, fourcc, 22.0, (W, H))

        status = "Waiting"
        rects = []

        # cv2.line(frame, (50, int(H // 2)), (W, int(H // 2)), (0, 0, 0), 2)

        # check to see if we should run a more computationally expensive
        # object detection method to aid our tracker
        if total_frames % skip_frames == 0:
            status = "Detecting"
            # Create a 4D blob from a frame.
            blob = cv2.dnn.blobFromImage(frame, 1 / 255, (inp_width, inp_height), [0, 0, 0], 1, crop=False)

            # Sets the input to the network
            net.setInput(blob)

            # Runs the forward pass to get output of the output layers
            outs = net.forward(get_outputs_names(net))

            # Remove the bounding boxes with low confidence and store in trackers list for future tracking
            trackers = fill_tracker_list(frame, outs, classes, conf_threshold, nms_threshold, rgb)

        # otherwise, we should utilize our object *trackers* rather than
        # object *detectors* to obtain a higher frame processing throughput
        else:
            # loop over the trackers
            for tracker in trackers:
                # set the status of our system to be 'tracking' rather
                # than 'waiting' or 'detecting'
                status = "Tracking"

                # If use opencv tracker
                # #update the tracker and grab the updated position
                # _, pos = tracker.update(rgb)
                #
                # # unpack the position object
                # startX = int(pos[0])
                # startY = int(pos[1])
                # endX = int(pos[2])
                # endY = int(pos[3])
                # #set the status of our system to be 'tracking' rather
                # #than 'waiting' or 'detecting'

                # update the tracker and grab the updated position
                tracker.update(rgb)
                pos = tracker.get_position()
                # unpack the position object
                start_x = int(pos.left())
                start_y = int(pos.top())
                end_x = int(pos.right())
                end_y = int(pos.bottom())
                # add the bounding box coordinates to the rectangles list

                cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (255, 255, 255), 3)
                rects.append((start_x, start_y, end_x, end_y))

        objects = ct.update(rects)

        # Mark  all the persons in the frame
        count = 0
        for (objectID, centroid) in objects.items():

            # check to see if a trackable object exists for the current
            # object ID
            to = trackableObjects.get(objectID, None)

            # if there is no existing trackable object, create one
            if to is None:
                to = TrackableObject(objectID, centroid)

            # otherwise, there is a trackable object so we can utilize it
            # to determine direction

            # otherwise, there is a trackable object so we can utilize it
            else:
                # the difference between the y-coordinate of the *current*
                # centroid and the mean of *previous* centroids will tell
                # us in which direction the object is moving (negative for
                # 'up' and positive for 'down')
                y = [c[1] for c in to.centroids]
                x_ = [c[0] for c in to.centroids]
                direction = centroid[1] - np.mean(y[:-2])
                direction_x = centroid[0]
                to.centroids.append(centroid)

                # check to see if the object has been counted or not

                if not to.counted:
                    # print(objectID, " ID")
                    # print(direction, " Direction  ", centroid[1], " centroid")
                    # print(centroid[1])
                    # print(centroid[0])
                    # if the direction is negative (indicating the object
                    # is moving up) AND the centroid is above the center
                    # line, count the object
                    if direction <= 0 and 1230 < centroid[0] < 1480 \
                            and 160 < centroid[1] < 265 and max(y) > 210:  # or mark your custom x and y val
                        print(objectID, " ID")
                        print(direction, " Direction  ", centroid[1], " centroid")
                        print(max(y))
                        print(centroid[1], ' ', centroid[0])
                        print("exit")
                        print("############")
                        total_up += 1
                        empty.append(total_up)
                        to.counted = True
                        print("----------------------------------------")
                    # if the direction is positive (indicating the object
                    # is moving down) AND the centroid is below the
                    # center line, count the object

                    if direction > 0 and 1200 < centroid[0] < 1490 \
                            and 130 < centroid[1] < 300 and min(y) < 400:
                        print(objectID, " ID")
                        print(direction, " Direction  ", centroid[1], " centroid")
                        print(min(y))
                        print(centroid[1], ' ', centroid[0])
                        print("Into")
                        print("############")
                        total_down += 1
                        empty1.append(total_down)
                        # print(empty1[-1])
                        x = []
                        # compute the sum of total people inside
                        x.append(len(empty1) - len(empty))

                        to.counted = True
                        print("----------------------------------------")
            # store the trackable object in our dictionary
            trackableObjects[objectID] = to

            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

            cv2.line(frame, (1200, 200), (1500, 225), (255, 0, 0), 10)
        # construct a tuple of information we will be displaying on the
        info = [
            ("Exit", total_up),
            ("Enter", total_down),
            ("Status", status),
        ]

        info2 = [
            ("Total people inside", x),
        ]

        # Display the output
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        for (i, (k, v)) in enumerate(info2):
            text = "{}: {}".format(k, v)
            if v and v[0] < 0:
                v[0] = 0
            cv2.putText(frame, text, (90, H - ((i * 20) + 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        writer.write(frame)
        cv2.imshow(win_name, cv2.resize(frame, (1600, 900)))
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        # increment the total number of frames processed thus far and
        # then update the FPS counter
        total_frames += 1
        fps.update()

    # stop the timer and display FPS information
    fps.stop()

    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    cv2.destroyAllWindows()


if __name__ == '__main__':
    run()
