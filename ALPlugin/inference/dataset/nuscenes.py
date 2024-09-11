
from pointcept.datasets.builder import DATASETS
from pointcept.datasets import NuScenesDataset

@DATASETS.register_module()
class DemoNuScenesDataset(NuScenesDataset):

    def get_data_list(self):

        return []