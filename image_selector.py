import ipywidgets as widgets
import cv2

from ndarray_image import NdarrayImage


class ImageSelector(widgets.HBox):

    def __init__(self, images, resize_to=(50, 100)):
        """
        Args:
            images: a list of ndarrays
            resize_to: a tuple (WIDTH, HEIGHT)
        """
        self.thumbs = [cv2.resize(img, resize_to) for img in images]
        self.active_mask = [True] * len(images)
        self.im_objs = [NdarrayImage(value=img) for img in self.thumbs]
        self.btns = [
            widgets.Button(description='RM',
                           button_style='success',
                           layout=widgets.Layout(width=f'{resize_to[0]}px'))
            for _ in range(len(images))
        ]
        for n, btn in enumerate(self.btns):
            btn.n = n
            btn.on_click(lambda x: self.button_callback(x))
        self.vboxes = [widgets.VBox([imo, btn]) for imo, btn in zip(self.im_objs, self.btns)]
        super().__init__(self.vboxes)

    def get_chosen(self):
        return self.active_mask

    def button_callback(self, b):
        n = b.n
        self.active_mask[n] ^= True  # toggle
        b.button_style = 'success' if self.active_mask[n] else 'danger'
