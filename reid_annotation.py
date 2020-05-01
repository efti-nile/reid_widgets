import itertools
import random
import os

import ipywidgets as widgets
import cv2

from ndarray_image import NdarrayImage


class IdRandomGen:

    def __init__(self, min=1000000, max=1999999):
        random.seed()
        self.min = min  # min int that can be generated
        self.max = max  # max int that can be generated
        self.ids = set()

    def get_id(self):
        candidate = random.randint(self.min, self.max)
        if candidate not in self.ids:
            self.ids.add(candidate)
            return candidate
        else:
            return self.get_id()  # try one more time


class ReidAnnotation(widgets.VBox):
    """Widget for manual annotation of a little amount of data"""

    class RandomImageBox(widgets.VBox):
        """Displays a track as a randomly chosen image with a selection button.
        Text on the button is the length of track
        """
        def __init__(self, imgs, resize_to=(50, 100)):
            """
            Args:
                imgs: a list of images in a track
                resize_to: a tuple (WIDTH, HEIGHT)
            """
            self.imgs = imgs
            self.thumb = cv2.resize(random.choice(self.imgs), resize_to)
            self.btn = widgets.Button(description=str(len(self.imgs)),
                                      button_style='success',
                                      layout=widgets.Layout(width=f'auto'))
            self.btn.on_click(self.button_callback)
            self.is_chosen = False  # if this image box selected by user
            self.im_obj = NdarrayImage(value=self.thumb)
            super().__init__([self.im_obj, self.btn])

        def button_callback(self, b):
            self.is_chosen ^= True
            b.button_style = 'danger' if self.is_chosen else 'success'

    def __init__(self, imgs_by_cam, dataset_name, datasets_root):
        """
        Args:
            imgs_by_cam: list[list[list[images]]] (imgs_by_cam[cam_idx][track_idx][img_idx])
        """
        self.imgs_by_cam = imgs_by_cam
        self.dataset_name = dataset_name
        self.datasets_root = datasets_root

        self.id_gen = IdRandomGen()
        self.btn = widgets.Button(description='NEXT', layout=widgets.Layout(width='60%', height='50px'))
        self.btn.on_click(self._button_callback)
        self.cam_box_layout = widgets.Layout(overflow='scroll hidden',
                                        border='3px solid black',
                                        width='500px',
                                        height='',
                                        flex_flow='row',
                                        display='flex')
        self._create_camboxes()
        super().__init__([self.btn] + self.cam_boxes)

    def _create_camboxes(self):
        self.cam_boxes = [
            widgets.Box(children=image_boxes, layout=self.cam_box_layout) for image_boxes in
            [[ReidAnnotation.RandomImageBox(imgs) for imgs in cam_imgs] for cam_imgs in self.imgs_by_cam]
        ]

    def _button_callback(self, b):
        reid_id = self.id_gen.get_id()
        # Get all chosen images
        selected_imgs_by_cam = {n: list(itertools.chain(*[img_box.imgs for img_box in cam_box.children if img_box.is_chosen]))
                                for n, cam_box in enumerate(self.cam_boxes)}
        # Save to `$dataset_root/$dataset_name/$cam_id/$reid_id/*.jpg`
        for cam_id, imgs in selected_imgs_by_cam.items():
            if imgs:
                dir_path = os.path.join(self.datasets_root, self.dataset_name, str(cam_id), str(reid_id))
                os.makedirs(dir_path, exist_ok=True)
                for img_id, img in enumerate(imgs):
                        cv2.imwrite(os.path.join(dir_path, f'{img_id}.jpg'), img)
        # Remove saved images from GUI
        for cambox in self.cam_boxes:
            cambox.children = [img_box for img_box in cambox.children if not img_box.is_chosen]
