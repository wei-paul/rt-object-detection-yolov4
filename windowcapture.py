import numpy as np
import win32gui
import win32ui
import win32con
from ctypes import windll
from PIL import Image
import os
from time import sleep


class WindowCapture:

    # Properties
    w = 0
    h = 0
    hwnd = None
    cropped_x = 0
    cropped_y = 0
    offset_x = 0
    offset_y = 0

    # Constructor
    def __init__(self, window_name=None):

        # Find the handle fo the window we want to capture.
        # If no window name is given, capture entire screen

        if window_name is None:
            self.hwnd = win32gui.GetDesktopWindow()
        else:
            self.hwnd = win32gui.FindWindow(None, window_name)
            if not self.hwnd:
                raise Exception('Window not found: {}'.format(window_name))

        # Get window size (to get rid of black spaces when capturing)
        window_rect = win32gui.GetWindowRect(self.hwnd)
        self.w = window_rect[2] - window_rect[0]
        self.h = window_rect[3] - window_rect[1]

        # Get rid of the inner window bar when capturing window
        border_pixels = 8
        titlebar_pixels = 30
        self.w = self.w - (border_pixels * 2)
        self.h = self.h - titlebar_pixels - border_pixels
        self.cropped_x = border_pixels
        self.cropped_y = titlebar_pixels

        # set the cropped coordinates offset so we can translate the screenshot (coordinates)
        # images into actual screen positions
        self.offset_x = window_rect[0] + self.cropped_x
        self.offset_y = window_rect[1] + self.cropped_y

    def get_screenshot(self):

        hwnd = self.hwnd
        left, top, right, bottom = win32gui.GetClientRect(hwnd)
        w = right - left
        h = bottom - top

        # Get device content
        hwnd_dc = win32gui.GetWindowDC(hwnd)
        # Create a compatible device context
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()

        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(mfc_dc, w, h)

        save_dc.SelectObject(bitmap)

        # If Special K is running, this number is 3. If not, 1
        # Used to capture window image, specifically designed to capture windows with hardware acceleration enabled
        # Or have certain rendering techniques. This provides a more reliable way to capture the window image compared to the BitBlt() method
        result = windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 3)

        bmpinfo = bitmap.GetInfo()
        bmpstr = bitmap.GetBitmapBits(True)
        # Using Pillow's Image here to ensure that the color channels are handled correctly based on the order returned by GetBitmapBits()
        # For future reference. If using library that expects the BGRA order, can use first approach just converting to Numpy Array using np.fromstring() without converting to PIL image object first.
        im = Image.frombuffer(
            "RGB", (bmpinfo["bmWidth"], bmpinfo["bmHeight"]), bmpstr, "raw", "BGRX", 0, 1)

        if result != 1:
            win32gui.DeleteObject(bitmap.GetHandle())
            save_dc.DeleteDC()
            mfc_dc.DeleteDC()
            win32gui.ReleaseDC(hwnd, hwnd_dc)
            raise RuntimeError(
                f"Unable to acquire screenshot! Result: {result}")

        img = np.array(im)[:, :, ::-1].copy()

        # Free Resources
        win32gui.DeleteObject(bitmap.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwnd_dc)

        # make image C_CONTIGUOUS to avoid errors that look like:
        # File ... in draw_rectangles
        # TypeERROR: an integer is required (got type tuple)
        img = np.ascontiguousarray(img)

        return img

    # Find the name of the window you're interested in. in FindWindow()

    def generate_image_dataset(self):
        if not os.path.exists("images"):
            os.mkdir("images")
        while (True):
            img = self.get_screenshot()
            im = Image.fromarray(img[..., [2, 1, 0]])
            im.save(f"./images/img_{len(os.listdir('images'))}.jpg")
            sleep(0.3)

    def get_window_size(self):
        return (self.w, self.h)

    @staticmethod
    def list_window_names():
        def winEnumHandler(hwnd, ctx):
            if win32gui.IsWindowVisible(hwnd):
                print(hex(hwnd), win32gui.GetWindowText(hwnd))
        win32gui.EnumWindows(winEnumHandler, None)

    # Translate a pixel position on a screenshot image to a pixel position on the screen.
    # pos = (x, y)
    # CAUTION: This will only return the correct position if you don't move the window being captured (after running the script)
    def get_screen_position(self, pos):
        return (pos[0] + self.offset_x, pos[1] + self.offset_y)
