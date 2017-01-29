# Returns true if the point is inside the polygon
# False otherwise


def in_polygon(x, y, xp, yp):
    c = 0
    for i in range(len(xp)):
        if (((yp[i] <= y < yp[i - 1]) or (yp[i-1] <= y < yp[i])) and
                (x > (xp[i-1] - xp[i]) * (y - yp[i]) / (yp[i-1] - yp[i]) + xp[i])):
            c = 1 - c
    return c == 1


# Return true if each pixel value is less than 40
# False otherwise

def threshold(pixel):
    if pixel[0] < 40 and pixel[1] < 40 and pixel[2] < 40:
        return False
    return True

# Return true if field pixel is green
# False otherwise


def on_field_background(hue):
    if hue in range(35, 50):
        return True
    return False

