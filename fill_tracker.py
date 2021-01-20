import cv2
import dlib
import numpy as np


# Remove the bounding boxes with low confidence using non-maxima suppression
def fill_tracker_list(frame, outs, classes, confThreshold, nmsThreshold, rgb):
    def draw_pred(class_id, conf, left, top, right, bottom):
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 255), 10)
        rectagle_center_pont = ((left + left + right - left) // 2, (top + top + bottom - top) // 2)
        cv2.circle(frame, rectagle_center_pont, 1, (0, 255, 0), 10)
        label = '%.2f' % conf

        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])

        text = "{}: {:.4f}".format(classes[class_id], confidences[val])
        cv2.putText(frame, text, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 255], 2)

    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    trackers = []

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            assert (class_id < len(classes))
            confidence = scores[class_id]
            # Check if confidence is more than threshold and the detected object is a person
            if confidence > confThreshold and classes and classes[class_id] == "person":
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for val in indices:
        val = val[0]
        box = boxes[val]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        tracker = dlib.correlation_tracker()
        rect = dlib.rectangle(left, top, left + width, top + height)
        tracker.start_track(rgb, rect)
        # cv2 tracker
        # tracker_2 = cv.TrackerKCF_create()
        # rect2 = left, top, left + width, top + height
        # tracker_2.init(frame, rect2)
        # add the tracker to our list of trackers so we can
        # utilize it during skip frames
        trackers.append(tracker)
        draw_pred(class_ids[val], confidences[val], left, top, left + width, top + height)
    return trackers
