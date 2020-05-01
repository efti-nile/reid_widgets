import io

import cv2
import numpy as np
import ipywidgets as widgets


class NdarrayImage(widgets.Image):
    def __init__(self, *args, **kwargs):
        bgr_img = kwargs['value']
        assert isinstance(bgr_img, np.ndarray)
        assert bgr_img.ndim == 3 and bgr_img.shape[2] == 3 and bgr_img.size
        is_success, buffer = cv2.imencode('.png', bgr_img)
        if not is_success:
            raise RuntimeError
        io_buf = io.BytesIO(buffer)
        kwargs['value'] = io_buf.read()
        kwargs['format'] = 'png'
        super(NdarrayImage, self).__init__(*args, **kwargs)
