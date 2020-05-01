import ipywidgets as widgets
from google.colab.patches import cv2_imshow
from IPython.display import clear_output
import cv2


class ImageSelector:
    """Image selector for Google Colab"""

    def __init__(self, images, resize_to=(50, 100)):
        """
        Args:
            images: a list of ndarrays
            resize_to: a tuple (WIDTH, HEIGHT)
        """
        self.thumbs = [cv2.resize(img, resize_to) for img in images]
        self.active_mask = [True] * len(images)
        self.outs = [widgets.Output() for _ in range(len(images))]
        self.btns = [
            widgets.Button(description='RM',
                           button_style='success',
                           layout=widgets.Layout(width=f'{resize_to[0]}px'))
            for _ in range(len(images))
        ]
        for n, btn in enumerate(self.btns):
            btn.n = n
            btn.on_click(lambda x: self.button_callback(x))
        self.vboxes = [widgets.VBox([out, btn]) for out, btn in zip(self.outs, self.btns)]
        for out, img in zip(self.outs, self.thumbs):
            with out:
                cv2_imshow(img)
        self.main_box = widgets.HBox(self.vboxes)

    def get_choosen(self):
        return self.active_mask

    def button_callback(self, b):
        n = b.n
        self.active_mask[n] ^= True  # toggle
        b.button_style = 'success' if self.active_mask[n] else 'danger'

    def __repr__(self):  # to work with display()
        display(self.main_box)
        return ''
