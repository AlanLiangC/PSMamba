"""
Main Testing Script

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import sys
sys.path.append('/data1/liangao/Projects/3D_Perception/3D_Segmentation/PSMamba')

from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from ALPlugin.inference.inference_engines import INFERENCERS
from pointcept.engines.launch import launch


def inference(data_dict):
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)
    cfg = default_setup(cfg)
    inferencer = INFERENCERS.build(dict(type='SemSegInferencer', cfg=cfg))
    inferencer.test_loader.dataset.data_list = [data_dict]
    inferencer.inference()

# def inference(data_dict):
#     args = default_argument_parser().parse_args()
#     cfg = default_config_parser(args.config_file, args.options)

#     launch(
#         inference_worker,
#         num_gpus_per_machine=args.num_gpus,
#         num_machines=args.num_machines,
#         machine_rank=args.machine_rank,
#         dist_url=args.dist_url,
#         cfg=(cfg, data_dict,),
#     )

if __name__ == "__main__":
    import pickle
    f = open('data/nuscenes/info/nuscenes_infos_10sweeps_val.pkl', 'rb')
    data_list = pickle.load(f)
    data_dict = data_list[0]
    inference(data_dict)
