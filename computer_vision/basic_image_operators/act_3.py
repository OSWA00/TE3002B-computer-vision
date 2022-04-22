import cv2 as cv
import numpy as np


def detect_shapes(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    edges = get_canny_edges(gray)

    cnts = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for cnt in cnts:
        approx = cv.approxPolyDP(cnt, 0.01 * cv.arcLength(cnt, True), True)
        if len(approx) == 5:
            print("Blue = pentagon")
            cv.drawContours(frame, [cnt], 0, 255, -1)
        elif len(approx) == 3:
            print("Green = triangle")
            cv.drawContours(frame, [cnt], 0, (0, 255, 0), -1)
        elif len(approx) == 4:
            print("Red = square")
            cv.drawContours(frame, [cnt], 0, (0, 0, 255), -1)
        elif len(approx) == 6:
            print("Cyan = Hexa")
            cv.drawContours(frame, [cnt], 0, (255, 255, 0), -1)
        elif len(approx) == 8:
            print("White = Octa")
            cv.drawContours(frame, [cnt], 0, (255, 255, 255), -1)
        elif len(approx) > 12:
            print("Yellow = circle")
            cv.drawContours(frame, [cnt], 0, (0, 255, 255), -1)

    return frame


def detect_lines(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    canny_edges = get_canny_edges(frame_gray)

    linesP = cv.HoughLinesP(canny_edges, 1, np.pi / 180, 50, None, 50, 10)

    try:
        for index, points in enumerate(linesP):
            lin = linesP[index][0]
            cv.line(
                frame, (lin[0], lin[1]), (lin[2], lin[3]), (0, 255, 0), 3, cv.LINE_AA
            )
    except:
        print("No lines found")
    return frame


def get_contours(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    canny_edges = get_canny_edges(frame_gray)

    contours, hierarchy = cv.findContours(
        image=canny_edges, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE
    )

    edges = np.zeros(frame.shape, dtype="uint8")
    cv.drawContours(
        image=edges,
        contours=contours,
        contourIdx=-1,
        color=(0, 255, 0),
        thickness=2,
        lineType=cv.LINE_AA,
    )

    frame_gray_copy = frame_gray.copy()
    frame_gray_copy = cv.cvtColor(frame_gray_copy, cv.COLOR_GRAY2BGR)

    return cv.addWeighted(frame_gray_copy, 0.5, edges, 1.0, 0.0)


def get_blobs(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gaussian = cv.GaussianBlur(frame_gray, (3, 3), 0)

    params = cv.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 200

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 50

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    detector = cv.SimpleBlobDetector_create(params)
    keypoints = detector.detect(frame_gaussian)
    im_with_keypoints = cv.drawKeypoints(
        frame_gray,
        keypoints,
        np.array([]),
        (0, 0, 255),
        cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    return im_with_keypoints


def get_canny_edges(frame):
    frame_gaussian = cv.GaussianBlur(frame, (5, 5), 0)
    frame_canny = cv.Canny(frame_gaussian, threshold1=100, threshold2=200)
    return frame_canny


if __name__ == "__main__":

    vid = cv.VideoCapture(0)

    detector = cv.SimpleBlobDetector_create()

    while True:

        ret, frame = vid.read()
        cv.imshow("frame", frame)

        cv.imshow("contours", get_contours(frame))

        cv.imshow("blobs", get_blobs(frame))

        cv.imshow("shapes", detect_shapes(frame))

        cv.imshow("lines", detect_lines(frame))

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    vid.release()

    cv.destroyAllWindows()
