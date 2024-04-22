
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
import sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
import os, glob
import smplx
import argparse
from tqdm import tqdm

from tools.objectmodel import ObjectModel
from tools.meshviewer import Mesh, MeshViewer, points2sphere, colors
from tools.utils import parse_npz
from tools.utils import params2torch
from tools.utils import makepath
from tools.utils import to_cpu
from tools.utils import euler
from tools.cfg_parser import Config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def render_sequences(cfg):

    grab_path = cfg.grab_path
    print(grab_path)
    # all_seqs = glob.glob(grab_path + '/*/*eat*.npz')
    all_seqs = glob.glob(grab_path + '/s10/*.npz')

    mv = MeshViewer(width=1600, height=1200,offscreen=True)


    # set the camera pose
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = euler([80, -15, 0], 'xzx')
    camera_pose[:3, 3] = np.array([-.5, -1.4, 1.5])
    mv.update_camera_pose(camera_pose)

    # Base Paths
    preds_base = './predicts'
    gt_base = './grab'
    preds_paths = os.listdir(preds_base)
    print(preds_paths)
    pred_seqs = []
    gt_seqs = []
    start_frames = []

    for i in preds_paths:
        data = i.split('_')
        if len(data) < 3:
            continue
        
        pred_seqs.append(os.path.join(preds_base, i))
        start_frames.append(int(data[-1][:-4]))
        gt_seqs.append(os.path.join(gt_base, data[0], f'{data[1]}_{data[2]}.npz'))

    assert len(gt_seqs) == len(pred_seqs)

    # choice = np.random.choice(len(all_seqs), 10, replace=False)
    # choice = [0, 1]
    # for i in tqdm(choice):
    #     vis_sequence(cfg,all_seqs[i], mv)

    for i in tqdm(range(len(gt_seqs))):
        vis_sequence_gt(cfg, gt_seqs[i], mv, (pred_seqs[i], start_frames[i]))
        vis_sequence_pred(cfg, gt_seqs[i], mv, (pred_seqs[i], start_frames[i]))
    mv.close_viewer()


def vis_sequence_gt(cfg, sequence, mv, pred):

        # seq_data = parse_npz(sequence)
        # pred_data = np.load(pred).reshape(60, -1)
        # n_comps = seq_data['n_comps']
        # gender = seq_data['gender']
        seq_data = np.load(sequence, allow_pickle=True)

        n_frames = seq_data['n_frames']
        body = seq_data['body'].item()
        sbj_id   = seq_data['sbj_id']
        framerate  = seq_data['framerate']
        gender   = seq_data['gender'].item()
        betas   = seq_data['betas']


        T = 60

        sbj_mesh = os.path.join(grab_path, '..', body['vtemp'])
        sbj_vtemp = np.array(Mesh(filename=sbj_mesh).vertices)
        sbj_parms = body['params']
        
        sbj_m = smplx.create(cfg.model_path, model_type='smplx',
                              gender=gender, ext='npz',
                              num_pca_comps=24,
                              create_global_orient=True,
                              create_body_pose=True,
                              create_betas=True,
                              create_left_hand_pose=True,
                              create_right_hand_pose=True,
                              create_expression=True,
                              create_jaw_pose=True,
                              create_leye_pose=True,
                              create_reye_pose=True,
                              create_transl=True,
                              batch_size=T,
                              v_template=sbj_vtemp
                              )
        
        for key in sbj_parms.keys():
            sbj_parms[key] = sbj_parms[key][pred[1]: pred[1] + T]
            # print(f'{key}: {sbj_parms[key].shape}')

        sbj_parms = params2torch(sbj_parms)
        verts_sbj = to_cpu(sbj_m(**sbj_parms).vertices)


        seq_render_path = makepath(sequence.replace('.npz','').replace(cfg.grab_path, cfg.gt_path))

        skip_frame = 1
        for frame in range(0,T, skip_frame):
            # o_mesh = Mesh(vertices=verts_obj[frame], faces=obj_mesh.faces, vc=colors['yellow'])
            # o_mesh.set_vertex_colors(vc=colors['red'], vertex_ids=seq_data['contact']['object'][frame] > 0)

            s_mesh = Mesh(vertices=verts_sbj[frame], faces=sbj_m.faces, vc=colors['pink'], smooth=True)
            # s_mesh.set_vertex_colors(vc=colors['red'], vertex_ids=seq_data['contact']['body'][frame] > 0)

            # s_mesh_wf = Mesh(vertices=verts_sbj[frame], faces=sbj_m.faces, vc=colors['grey'], wireframe=True)
            # t_mesh = Mesh(vertices=verts_table[frame], faces=table_mesh.faces, vc=colors['white'])

            # mv.set_static_meshes([o_mesh, s_mesh, s_mesh_wf, t_mesh])
            mv.set_static_meshes([s_mesh])
            mv.save_snapshot(seq_render_path+'/%04d.png'%frame)


def vis_sequence_pred(cfg, sequence, mv, pred):
        
        pred_data = np.load(pred[0]).reshape(60, -1)
        seq_data = np.load(sequence, allow_pickle=True)

        n_frames = seq_data['n_frames']
        body = seq_data['body'].item()
        sbj_id   = seq_data['sbj_id']
        framerate  = seq_data['framerate']
        gender   = seq_data['gender'].item()
        betas   = seq_data['betas']


        T = 60

        sbj_mesh = os.path.join(grab_path, '..', body['vtemp'])
        sbj_vtemp = np.array(Mesh(filename=sbj_mesh).vertices)
        sbj_parms = body['params']

        # sbj_m = smplx.create(model_path=cfg.model_path,
        #                      model_type='smplh',
        #                      gender=gender,
        #                      num_pca_comps=n_comps,
        #                      v_template=sbj_vtemp,
        #                      use_pca = False,
        #                      batch_size=T)
        
        sbj_m = smplx.create(cfg.model_path, model_type='smplx',
                              gender=gender, ext='npz',
                              use_pca=False,
                              create_global_orient=True,
                              create_body_pose=True,
                              create_betas=True,
                              create_left_hand_pose=True,
                              create_right_hand_pose=True,
                              create_expression=True,
                              create_jaw_pose=True,
                              create_leye_pose=True,
                              create_reye_pose=True,
                              create_transl=True,
                              batch_size=T,
                              v_template=sbj_vtemp
                              )
        
        for key in sbj_parms.keys():
            sbj_parms[key] = sbj_parms[key][pred[1]: pred[1] + T]
            # print(f'{key}: {sbj_parms[key].shape}')

        sbj_parms['body_pose'] = pred_data[:, :63]
        sbj_parms['left_hand_pose'] = pred_data[:, 63: 63+45]
        sbj_parms['right_hand_pose'] = pred_data[:, 63+45:]

        sbj_parms = params2torch(sbj_parms)
        verts_sbj = to_cpu(sbj_m(**sbj_parms).vertices)

        seq_render_path = makepath(sequence.replace('.npz','').replace(cfg.grab_path, cfg.pred_path))

        skip_frame = 1
        for frame in range(0,T, skip_frame):
            # o_mesh = Mesh(vertices=verts_obj[frame], faces=obj_mesh.faces, vc=colors['yellow'])
            # o_mesh.set_vertex_colors(vc=colors['red'], vertex_ids=seq_data['contact']['object'][frame] > 0)

            s_mesh = Mesh(vertices=verts_sbj[frame], faces=sbj_m.faces, vc=colors['pink'], smooth=True)
            # s_mesh.set_vertex_colors(vc=colors['red'], vertex_ids=seq_data['contact']['body'][frame] > 0)

            # s_mesh_wf = Mesh(vertices=verts_sbj[frame], faces=sbj_m.faces, vc=colors['grey'], wireframe=True)
            # t_mesh = Mesh(vertices=verts_table[frame], faces=table_mesh.faces, vc=colors['white'])

            # mv.set_static_meshes([o_mesh, s_mesh, s_mesh_wf, t_mesh])
            mv.set_static_meshes([s_mesh])
            mv.save_snapshot(seq_render_path+'/%04d.png'%frame)


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='GRAB-render')

    parser.add_argument('--grab-path', required=True, type=str,
                        help='The path to the downloaded grab data')
    parser.add_argument('--render-path', required=True, type=str,
                        help='The path to the folder to save the renderings')
    parser.add_argument('--model-path', required=True, type=str,
                        help='The path to the folder containing smplx models')

    args = parser.parse_args()

    grab_path = args.grab_path
    render_path = args.render_path
    model_path = args.model_path

    # grab_path = 'PATH_TO_DOWNLOADED_GRAB_DATA/grab'
    # render_path = 'PATH_TO_THE LOCATION_TO_SAVE_RENDERS'
    # model_path = 'PATH_TO_DOWNLOADED_MODELS_FROM_SMPLX_WEBSITE/'

    cfg = {
        'grab_path': grab_path,
        'model_path': model_path,
        'render_path':render_path,
        'gt_path': os.path.join(render_path, 'gt'),
        'pred_path': os.path.join(render_path, 'pred')
    }

    cfg = Config(**cfg)
    render_sequences(cfg)

