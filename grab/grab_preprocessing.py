
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
from tools.cfg_parser import Config
from tools.utils import makepath, makelogger
from tools.meshviewer import Mesh
from tools.utils import parse_npz
from tools.utils import params2torch
from tools.utils import prepare_params
from tools.utils import to_cpu
from tools.utils import append2dict
from tools.utils import np2torch
from plyfile import PlyData
from bps import bps
# from bps_torch.bps import bps_torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INTENTS = ['lift', 'pass', 'offhand', 'use', 'all']

OBJECT_MESH_PATH='/cluster/courses/digital_humans/datasets/team_1/GRAB/tools/object_meshes/contact_meshes'

def get_bps_from_path(obj_name):

    object_name=obj_name+'.npy'
    object_path=os.path.join(OBJECT_MESH_PATH, object_name)

    return np.load(object_path)

class GRABDataSet(object):

    def __init__(self, cfg, logger=None, **params):

        self.cfg = cfg
        self.grab_path = cfg.grab_path
        self.out_path = cfg.out_path
        makepath(self.out_path)

        if logger is None:
            log_dir = os.path.join(self.out_path, 'grab_preprocessing.log')
            self.logger = makelogger(log_dir=log_dir, mode='a').info
        else:
            self.logger = logger
        self.logger('Starting data preprocessing !')

        assert cfg.intent in INTENTS

        self.intent = cfg.intent
        self.logger('intent:%s --> processing %s sequences!' % (self.intent, self.intent))

        if cfg.splits is None:
            self.splits = {'test': ['mug', 'wineglass', 'camera', 'binoculars', 'fryingpan', 'toothpaste'],
                            'val': ['apple', 'toothbrush', 'elephant', 'hand'],
                            'train': ['hand']}
        else:
            assert isinstance(cfg.splits, dict)
            self.splits = cfg.splits
            
        self.all_seqs = glob.glob(self.grab_path + '/*/*.npz')
        
        ## to be filled 
        self.selected_seqs = []
        self.obj_based_seqs = {}
        self.sbj_based_seqs = {}
        self.split_seqs = {'test': [],
                           'val': [],
                           'train': []
                           }

        # group, mask, and sort sequences based on objects, subjects, and intents
        self.process_sequences()

        self.logger('Total sequences: %d' % len(self.all_seqs))
        self.logger('Selected sequences: %d' % len(self.selected_seqs))
        self.logger('Number of sequences in each data split : train: %d , test: %d , val: %d'
                         %(len(self.split_seqs['train']), len(self.split_seqs['test']), len(self.split_seqs['val'])))
        self.logger('Number of objects in each data split : train: %d , test: %d , val: %d'
                         % (len(self.splits['train']), len(self.splits['test']), len(self.splits['val'])))

        # process the data
        self.data_preprocessing(cfg)

    def data_preprocessing(self,cfg):


        self.subject_mesh = {}
        self.obj_info = {}
        self.sbj_info = {}

        min_len = 1e5
        max_len = 0

        for split in self.split_seqs.keys():
        # for split in ['train']:

            self.logger('Processing data for %s split.' % (split))

            frame_names = []
            body_data = {
                'global_orient': [],'body_pose': [],'transl': [],
                'right_hand_pose': [],'left_hand_pose': [],
                'jaw_pose': [],'leye_pose': [],'reye_pose': [],
                'expression': [],'fullpose': [],
                'contact':[], 'verts' :[]
            }

            object_data ={'verts': [], 'global_orient': [], 'transl': [], 'contact': []}
            lhand_data = {'verts': [], 'global_orient': [], 'hand_pose': [], 'transl': [], 'fullpose': []}
            rhand_data = {'verts': [], 'global_orient': [], 'hand_pose': [], 'transl': [], 'fullpose': []}
            if not os.path.exists(os.path.join(cfg.out_path, split)):
                os.mkdir(os.path.join(cfg.out_path, split))

            split_len = 0
            # for sequence in tqdm(self.split_seqs[split]):
            for sequence in tqdm(self.split_seqs[split]):

                seq_data = parse_npz(sequence)

                obj_name = seq_data.obj_name
                sbj_id   = seq_data.sbj_id
                n_comps  = seq_data.n_comps
                gender   = seq_data.gender

                frame_mask = self.filter_contact_frames(seq_data)
                all_frames = len(seq_data['contact']['object'])
                contact_frames = (seq_data['contact']['object'] >0).sum()
                valid_frames = (seq_data['contact']['object'] >-1).sum()

                mesh_path = os.path.abspath(f'./tools/subject_meshes/{gender}/{sbj_id}.ply')


                if all_frames > frame_mask.sum():
                    print("Lost Frames")

                # frame_mask = np.ones((n_frames,))
                # total selectd frames
                T = frame_mask.sum()
                if T < 300:
                    continue # if no frame is selected continue to the next sequence
                split_len += 1
                
                # print(type(seq_data.body.item()))
                sbj_params = prepare_params(seq_data.body.params, frame_mask)
                rh_params  = prepare_params(seq_data.rhand.params, frame_mask)
                lh_params  = prepare_params(seq_data.lhand.params, frame_mask)
                obj_params = prepare_params(seq_data.object.params, frame_mask)


                file_names = sequence.split('/')

                # keys = sbj_params.keys()
                # for key in keys:
                #     print(f'{key}: {sbj_params[key].shape}')

                """
                The pose details
                # transl: (299, 3)
                # global_orient: (299, 3)
                # body_pose: (299, 63): 21 JOINTS
                # jaw_pose: (299, 3)
                # leye_pose: (299, 3)
                # reye_pose: (299, 3)
                # left_hand_pose: (299, 24) : 8 JOINTS
                # right_hand_pose: (299, 24)
                # fullpose: (299, 165)
                # expression: (299, 10)
                """

                global_orient = sbj_params["global_orient"]
                # print(global_orient.shape)
                root_transl = sbj_params["transl"]
                # print(root_transl.shape)
                body_pose = sbj_params["body_pose"]
                # print(body_pose.shape)
                lhand_pose = lh_params["fullpose"]
                # print(lhand_pose.shape)
                rhand_pose = rh_params["fullpose"]
                # print(rhand_pose.shape)
                obj_orient = obj_params["global_orient"]
                # print(obj_orient.shape)
                obj_transl = obj_params["transl"]
                # print(obj_transl.shape)
                obj_bps = get_bps_from_path(obj_name)
                # print(obj__mesh_verts.shape)
                
                max_len = max(max_len, T)
                min_len = min(min_len, T)

                sbj_save = {}
                sbj_save['pose'] = np.concatenate([global_orient, body_pose, lhand_pose, rhand_pose], axis = 1)
                sbj_save['trans'] = root_transl
                sbj_save['betas'] = np.load(f'tools/subject_meshes/{gender}/{sbj_id}_betas.npy')
                sbj_save['gender'] = gender
                sbj_save['obj_orient'] = obj_orient
                sbj_save['obj_transl'] = obj_transl
                sbj_save['obj_contact'] = object_data['contact']
                sbj_save['obj_verts'] = object_data['verts']
                sbj_save['obj_bps'] = obj_bps
                sbj_save['contact_frames'] = frame_mask
                sbj_save['mesh_path'] = mesh_path
                sbj_save['n_comps'] = n_comps
                sbj_save['sbj_params']=sbj_params
                np.save(os.path.join(self.out_path, split, f'{file_names[-2]}_{file_names[-1]}.npy'), sbj_save)
                # print(sbj_save.shape)

            print(f"{split} len: {split_len}")

        print(f'MAX_LEN {max_len}')
        print(f'MIN_LEN {min_len}')

    def process_sequences(self):

        for sequence in self.all_seqs:
            subject_id = sequence.split('/')[-2]
            action_name = os.path.basename(sequence)
            object_name = action_name.split('_')[0]

            # filter data based on the motion intent
            if self.intent == 'all':
                pass
            elif self.intent == 'use' and any(intnt in action_name for intnt in INTENTS[:3]):
                continue
            elif self.intent not in action_name:
                continue

            # group motion sequences based on objects
            if object_name not in self.obj_based_seqs:
                self.obj_based_seqs[object_name] = [sequence]
            else:
                self.obj_based_seqs[object_name].append(sequence)

            # group motion sequences based on subjects
            if subject_id not in self.sbj_based_seqs:
                self.sbj_based_seqs[subject_id] = [sequence]
            else:
                self.sbj_based_seqs[subject_id].append(sequence)

            # split train, val, and test sequences
            self.selected_seqs.append(sequence)
            if object_name in self.splits['test']:
                self.split_seqs['test'].append(sequence)
            elif object_name in self.splits['val']:
                self.split_seqs['val'].append(sequence)
            else:
                self.split_seqs['train'].append(sequence)
                if object_name not in self.splits['train']:
                    self.splits['train'].append(object_name)

    def filter_contact_frames(self, seq_data):
        if self.cfg.only_contact:
            frame_mask = (seq_data['contact']['object']>0).any(axis=1)
        else:
            frame_mask = (seq_data['contact']['object']>-1).any(axis=1)
        return frame_mask

    def load_obj_verts(self, obj_name, seq_data, n_verts_sample=512):

        mesh_path = os.path.join(self.grab_path, '..',seq_data.object.object_mesh)
        if obj_name not in self.obj_info:
            np.random.seed(100)
            obj_mesh = Mesh(filename=mesh_path)
            verts_obj = np.array(obj_mesh.vertices)
            faces_obj = np.array(obj_mesh.faces)

            if verts_obj.shape[0] > n_verts_sample:
                verts_sample_id = np.random.choice(verts_obj.shape[0], n_verts_sample, replace=False)
            else:
                verts_sample_id = np.arange(verts_obj.shape[0])

            verts_sampled = verts_obj[verts_sample_id]
            self.obj_info[obj_name] = {'verts': verts_obj,
                                       'faces': faces_obj,
                                       'verts_sample_id': verts_sample_id,
                                       'verts_sample': verts_sampled,
                                       'obj_mesh_file': mesh_path}

        return self.obj_info[obj_name]

    def load_sbj_verts(self, sbj_id, seq_data):

        mesh_path = os.path.join(self.grab_path, '..',seq_data.body.vtemp)
        if sbj_id in self.sbj_info:
            sbj_vtemp = self.sbj_info[sbj_id]
        else:
            sbj_vtemp = np.array(Mesh(filename=mesh_path).vertices)
            self.sbj_info[sbj_id] = sbj_vtemp
        return sbj_vtemp


if __name__ == '__main__':


    instructions = ''' 
    Please do the following steps before starting the GRAB dataset processing:
    1. Download GRAB dataset from the website https://grab.is.tue.mpg.de/ 
    2. Set the grab_path, out_path to the correct folder
    3. Change the configuration file for your desired data, like:
    
        a) if you only need the frames with contact,
        b) if you need body, hand, or object vertices to be computed,
        c) which data splits
            and etc
        
        WARNING: saving vertices requires a high-capacity RAM memory.
        
    4. In case you need body or hand vertices make sure to set the model_path
        to the models downloaded from smplx website 
    
    This code will process the data and save the pt files in the out_path folder.
    You can use the dataloader.py file to load and use the data.    

        '''

    parser = argparse.ArgumentParser(description='GRAB-vertices')

    parser.add_argument('--grab-path', required=True, type=str,
                        help='The path to the downloaded grab data')
    parser.add_argument('--out-path', default=None, type=str,
                        help='The path to the folder to save the processed data')
    parser.add_argument('--model-path', required=True, type=str,
                        help='The path to the folder containing smplx models')

    args = parser.parse_args()

    grab_path = args.grab_path
    out_path = args.out_path
    model_path = args.model_path
    process_id = 'GRAB_HML' # choose an appropriate ID for the processed data


    # grab_path = 'PATH_TO_DOWNLOADED_GRAB_DATA/grab'
    # out_path = 'PATH_TO_THE LOCATION_TO_SAVE_DATA'
    # process_id = 'GRAB_V00' # choose an appropriate ID for the processed data
    # model_path = 'PATH_TO_DOWNLOADED_MODELS_FROM_SMPLX_WEBSITE/'


    # split the dataset based on the objects
    grab_splits = { 'test': ['mug', 'wineglass', 'camera', 'binoculars', 'fryingpan', 'toothpaste'],
                    'val': ['apple', 'toothbrush', 'elephant', 'hand'],
                    'train': []}

    # split the dataset based on the subjects
    # grab_splits = { 'test': ['s10', 's9'],
    #                 'val': ['s8'],
    #                 'train': []}


    cfg = {

        'intent':'all', # from 'all', 'use' , 'pass', 'lift' , 'offhand'
        'only_contact':False, # if True, returns only frames with contact
        'save_body_verts': False, # if True, will compute and save the body vertices
        'save_lhand_verts': False, # if True, will compute and save the body vertices
        'save_rhand_verts': False, # if True, will compute and save the body vertices
        'save_object_verts': False,

        'save_contact': True, # if True, will add the contact info to the saved data

        # splits
        'splits':grab_splits,

        #IO path
        'grab_path': grab_path,
        'out_path': os.path.join(out_path, process_id),

        # number of vertices samples for each object
        'n_verts_sample': 1024,

        # body and hand model path
        'model_path':model_path,
    }

    # print(f'Outpath:{os.path.join(out_path, process_id)}')

    cwd = os.getcwd()
    default_cfg_path = os.path.join(cwd, '../configs/grab_preprocessing_cfg.yaml')
    cfg = Config(default_cfg_path=default_cfg_path, **cfg)
    cfg.write_cfg(write_path=cfg.out_path+'/grab_preprocessing_cfg.yaml')

    log_dir = os.path.join(cfg.out_path, 'grab_processing.log')
    logger = makelogger(log_dir=log_dir, mode='a').info

    logger(instructions)

    GRABDataSet(cfg, logger)

