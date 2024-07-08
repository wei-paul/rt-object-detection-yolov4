import cv2 as cv
import numpy as np


# provide a wholistic image, and an object (needle) you want to find in that image (haystack)
def findClickPosition(needle_img_path, haystack_img_path, threshold=0.80, debug_mode=None):

    haystack_img = cv.imread(haystack_img_path, cv.IMREAD_REDUCED_COLOR_2)
    needle_img = cv.imread(needle_img_path, cv.IMREAD_REDUCED_COLOR_2)

    needle_w = needle_img.shape[1]
    needle_h = needle_img.shape[0]

    # The matchTemplate will find the best matching object in the image
    # Returns a matrix of arrays of confidence scores (trying to match from x=0, y=0, to the end)
    method = cv.TM_CCOEFF_NORMED
    result = cv.matchTemplate(haystack_img, needle_img, method)

    # This will return array of all the matching object's X positions and  positions
    # For example array([421, 421, 422, 422, 422]) means there are 5 objects (numbers are positions at y values, then x values)
    locations = np.where(result >= threshold)

    # Super confusing format, so we convert it into (x, y) tuples
    # Explanation of the code from inside out: ::1 reverses the list (i.e [[10, 20, 30], [1, 2, 3]] > [[1,2,3], [10,20,30]])
    # * unpacks the list and zip merges list into new list of each item at the same index
    locations = list(zip(*locations[::-1]))

    # print("all location > 0.85: ", locations)

    # first we need to create the list of [x, y, w, h] rectangles
    rectangles = []

    for loc in locations:
        rect = [int(loc[0]), int(loc[1]), needle_w, needle_h]
        rectangles.append(rect)
        rectangles.append(rect)

    rectangles, weights = cv.groupRectangles(rectangles, 1, 0.5)

    points = []
    if len(rectangles):
        print("Found needle.")

        line_color = (0, 255, 0)
        line_type = cv.LINE_4
        marker_color = (255, 0, 255)
        marker_type = cv.MARKER_CROSS

        # need to loop over all the locations and draw their rectangles
        for (x, y, w, h) in rectangles:
            # Determine the center position and draw crosshair in found result
            center_x = x + int(w/2)
            center_y = y + int(h/2)
            # save the points
            points.append((center_x, center_y))

            if debug_mode == 'rectangles':
                # Draw rectangles around found results
                # determine the box positions
                top_left = (x, y)
                bottom_right = (x + w, y + h)
                # draw the box
                cv.rectangle(haystack_img, top_left,
                             bottom_right, line_color, line_type)
            elif debug_mode == 'points':
                cv.drawMarker(haystack_img, (center_x, center_y),
                              marker_color, marker_type)
        if debug_mode:
            cv.imshow('Matches', haystack_img)
            cv.waitKey()

    return points


points = findClickPosition(
    'syndrasmall.png', 'board.png', threshold=0.7, debug_mode='rectangles')
print(points)
