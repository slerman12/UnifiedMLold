# Copyright Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from pathlib import Path
import imageio  # M1 Mac: comment out freeimage imports in imageio/plugins/_init_


class Vlogger:
    def __init__(self, root_path, fps):
        self.save_path = Path(root_path) / 'Video'
        self.save_path.mkdir(exist_ok=True)
        self.fps = fps

    def vlog(self, vlogs, name):
        path = self.save_path / name
        imageio.mimsave(str(path), vlogs, fps=self.fps)
