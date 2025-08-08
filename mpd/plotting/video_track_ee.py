# import the necessary packages
import os

from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2

from mpd.models import default

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-v",
        "--video",
        type=str,
        # default="trajectory_000.mp4",
        default="/mnt/ssd2/carvalho/Projects/MotionPlanningDiffusion/logs_inference_real_robot/"
        "HumanDemos_v00/extra_object_2_boxes__TRUE/mpd/10/trajectory-000-004.mp4",
        help="path to input video file",
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=False,
    )
    ap.add_argument("-t", "--tracker", type=str, default="csrt", help="OpenCV object tracker type")
    args = vars(ap.parse_args())

    # extract the OpenCV version info
    (major, minor) = cv2.__version__.split(".")[:2]
    # if we are using OpenCV 3.2 OR BEFORE, we can use a special factory
    # function to create our object tracker
    if int(major) == 3 and int(minor) < 3:
        tracker = cv2.Tracker_create(args["tracker"].upper())
    # otherwise, for OpenCV 3.3 OR NEWER, we need to explicity call the
    # appropriate object tracker constructor:
    else:
        # initialize a dictionary that maps strings to their corresponding
        # OpenCV object tracker implementations
        OPENCV_OBJECT_TRACKERS = {
            "csrt": cv2.TrackerCSRT_create,
            "kcf": cv2.TrackerKCF_create,
            # "boosting": cv2.TrackerBoosting_create,
            "mil": cv2.TrackerMIL_create,
            # "tld": cv2.TrackerTLD_create,
            # "medianflow": cv2.TrackerMedianFlow_create,
            # "mosse": cv2.TrackerMOSSE_create
        }
        # grab the appropriate object tracker using our dictionary of
        # OpenCV object tracker objects
        tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
    # initialize the bounding box coordinates of the object we are going
    # to track
    initBB = None

    # if a video path was not supplied, grab the reference to the webcam
    if not args.get("video", False):
        print("[INFO] starting video stream...")
        vs = VideoStream(src=0).start()
        time.sleep(1.0)
    # otherwise, grab a reference to the video file
    else:
        vs = cv2.VideoCapture(args["video"])
    # initialize the FPS throughput estimator
    fps = None

    # video writer
    writer_W = 1280
    writer_H = 720
    output_video_path = args["video"].replace(".mp4", "_tracked.mp4")
    if args["output_dir"] is not None:
        os.makedirs(args["output_dir"], exist_ok=True)
        output_video_path = os.path.join(args["output_dir"], output_video_path.split("/")[-1])
    result = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (writer_W, writer_H),
    )

    box_center_l = []
    # Radius of circle
    radius = 4
    # Red color in BGR
    # color = (0, 0, 255)
    color = (0, 140, 255)
    # Line thickness of -1 px
    thickness = -1

    # loop over frames from the video stream
    i = 0
    while True:
        # grab the current frame, then handle if we are using a
        # VideoStream or VideoCapture object
        frame = vs.read()
        frame = frame[1] if args.get("video", False) else frame
        # check to see if we have reached the end of the stream
        if frame is None:
            break
        # resize the frame (so we can process it faster) and grab the
        # frame dimensions
        frame = imutils.resize(frame, width=writer_W)
        (H, W) = frame.shape[:2]

        # check to see if we are currently tracking an object
        if initBB is not None:
            # grab the new bounding box coordinates of the object
            (success, box) = tracker.update(frame)
            # check to see if the tracking was a success
            if success:
                (x, y, w, h) = [int(v) for v in box]
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # draw a circle in the center of the bounding box
                p1 = (x, y)
                p2 = (x + w, y + h)
                box_center = (int(p1[0] + p2[0]) // 2, int(p1[1] + p2[1]) // 2)
                box_center_l.append(box_center)
            # update the FPS counter
            fps.update()
            fps.stop()
            # initialize the set of information we'll be displaying on
            # the frame
            info = [
                ("Tracker", args["tracker"]),
                ("Success", "Yes" if success else "No"),
                ("FPS", "{:.2f}".format(fps.fps())),
            ]
            # loop over the info tuples and draw them on our frame
            for i, (k, v) in enumerate(info):
                text = "{}: {}".format(k, v)
                # cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                # cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            # draw the predicted bounding box center position
            for box_center in box_center_l:
                cv2.circle(frame, (box_center[0], box_center[1]), radius, color, thickness)

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        # key = cv2.waitKey(0)
        # if the 's' key is selected, we are going to "select" a bounding
        # box to track
        if key == ord("s") or i == 0:
            # select the bounding box of the object we want to track (make
            # sure you press ENTER or SPACE after selecting the ROI)
            initBB = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
            # start OpenCV object tracker using the supplied bounding box
            # coordinates, then start the FPS throughput estimator as well
            tracker.init(frame, initBB)
            fps = FPS().start()
        elif key == ord("q"):
            # if the `q` key was pressed, break from the loop
            break

        # write the frame
        result.write(frame)

    # if we are using a webcam, release the pointer
    if not args.get("video", False):
        vs.stop()
    # otherwise, release the file pointer
    else:
        vs.release()
        result.release()

    # close all windows
    cv2.destroyAllWindows()
