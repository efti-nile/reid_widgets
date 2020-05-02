import ipywidgets as widgets
from IPython.display import clear_output
import cv2

from image_selector import ImageSelector


NUM_IMGS_TO_SHOW = 10


class ReidChecker:
    def __init__(self, reid_dict, cam_num, get_imgs_func, refine_func):
        """
        Args:
            reid_dict: Dict[reid_ID] -> ( Dict[cam_id] -> Object )
            cam_num: total number of cameras
            get_imgs_func: a function Object -> List[ndarray images]
            refine_func: a function (Object, selection mask) -> Object
        """
        self.reid_dict = reid_dict
        self.cam_num = cam_num
        self.get_imgs = get_imgs_func
        self.refine = refine_func

        self.ids_to_sort = list(self.reid_dict.keys())  # a list with IDs
        self.ids_ok = []
        self.ids_skip = []
        self.cur_id = None

        self.img_selectors = None

        self.is_finalized = False
        self.tblists_ok = tuple([] for _ in range(self.cam_num))
        self.tblists_skip = tuple([] for _ in range(self.cam_num))

        self.outs = [widgets.Output() for _ in range(2)]

        self.btn_ok = widgets.Button(
            description='OK',
            layout=widgets.Layout(width='50%', height='50px')
        )
        self.btn_skip = widgets.Button(
            description='SKIP',
            layout=widgets.Layout(width='50%', height='50px')
        )
        self.btn_ok.on_click(lambda b: self.ok_callback(b))
        self.btn_skip.on_click(lambda b: self.skip_callback(b))

        with self.outs[0]:
            display(widgets.HBox([self.btn_ok, self.btn_skip]))

        self.next_id()

    def next_id(self):
        with self.outs[1]:
            clear_output()
            if self.ids_to_sort:
                self.cur_id = self.ids_to_sort.pop(0)
                self.img_selectors = []
                for tb in self.reid_dict[self.cur_id].values():
                    imgs = self.get_imgs(tb)[:NUM_IMGS_TO_SHOW]
                    print(f'ID: {self.cur_id} Loading {len(imgs)} images...')
                    cis = ImageSelector(imgs)
                    self.img_selectors.append(cis)
                    display(cis)
            else:
                print('No more data')
                self.finalize()

    def ok_callback(self, b):
        self.ids_ok.append(self.cur_id)
        self.apply_selector()
        self.next_id()

    def skip_callback(self, b):
        self.ids_skip.append(self.cur_id)
        self.apply_selector()
        self.next_id()

    def apply_selector(self):
        for tb, slt in zip(self.reid_dict[self.cur_id].values(), self.img_selectors):
            self.refine(tb, slt.get_chosen())

    def finalize(self):
        total = len(self.ids_skip) + len(self.ids_ok)
        percs = round(len(self.ids_ok) * 100.0 / total)
        print(f'Total:   {total}\nCorrect: {len(self.ids_ok)} ({percs}%)')
        self.btn_ok.disabled = True
        self.btn_skip.disabled = True
        self.make_out()
        self.is_finalized = True

    def make_out(self):
        for id in self.ids_ok:
            for cam_id, tb in self.reid_dict[id].items():
                self.tblists_ok[cam_id].append(tb)
        for id in self.ids_skip:
            for cam_id, tb in self.reid_dict[id].items():
                self.tblists_skip[cam_id].append(tb)

    def __repr__(self):
        for out in self.outs:
            display(out)
        return ''
