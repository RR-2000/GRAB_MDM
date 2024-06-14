import sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import os, glob
import argparse

from tqdm import tqdm
from plyfile import PlyData
from bps import bps

OBJECT_MESH_PATH='/cluster/courses/digital_humans/datasets/team_1/GRAB/tools/object_meshes/contact_meshes'

def get_object_from_path(obj_name):

    object_name=obj_name+'.ply'
    object_path=os.path.join(OBJECT_MESH_PATH, object_name)
    
    ply = PlyData.read(object_path)
    base_vertices = np.array(
        [(vertex["x"], vertex["z"],vertex["y"] ) for vertex in ply["vertex"]]
    )
    return base_vertices


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GRAB-vertices')

    parser.add_argument('--bps_dim', required=False, type=int,
                        help='The reduced dimension of BPS encoder', default = 1024)

    args = parser.parse_args()

    BPS_DIM = args.bps_dim

    object_list = [x for x in os.listdir(OBJECT_MESH_PATH) if x[-4:] == ".ply"]

    for obj_idx in tqdm(range(len(object_list))):
        obj__mesh_verts = get_object_from_path(object_list[obj_idx][:-4])
        bps_enc = bps.encode(np.expand_dims(obj__mesh_verts,0), bps_arrangement='grid', n_bps_points=BPS_DIM, bps_cell_type='dists').reshape(-1)
        np.save(os.path.join(OBJECT_MESH_PATH, object_list[obj_idx][:-4]), bps_enc)