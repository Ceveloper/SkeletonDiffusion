import torch
import numpy as np
from .base import Kinematic

#####################################################################################################################
#### Human3.6M joint representation
#####################################################################################################################

# for human36m. Dataset gives 32 jionts, with duplicates
JOINTS_DICT_H36M_32 = {0: "Hip",
                       1: "RHip", 2: "RKnee", 3: "RAnkle", 4: "RFoot", 5: "RToes",
                       6: "LHip", 7: "LKnee", 8: "LAnkle", 9: "LFoot", 10: "LToes",
                       11: "Hip",
                       12: "Torso", 13: "Neck", 14: "Nose", 15: "Head",
                       16: "Nose",
                       17: "LShoulder", 18: "LElbow", 19: "LWrist",
                       20: "LHand", 21: "LSmallFinger", 22: "LCenterFinger", 23: "LThumb",
                       24: "Nose",
                       25: "RShoulder", 26: "RElbow", 27: "RWrist",
                       28: "RHand", 29: "RSmallFinger", 30: "RCenterFinger", 31: "RThumb", }

# representation without duplicates. 25 joints.
CONVERSION_IDX_H36M_32TO25 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 21, 22, 25, 26, 27, 29, 30]
JOINTS_DICT_H36M_25 = {0: "Hip",
                       1: "RHip", 2: "RKnee", 3: "RAnkle", 4: "RFoot", 5: "RToes",
                       6: "LHip", 7: "LKnee", 8: "LAnkle", 9: "LFoot", 10: "LToes",
                       11: "Torso", 12: "Neck", 13: "Nose", 14: "Head",
                       15: "LShoulder", 16: "LElbow", 17: "LWrist",
                       18: "LSmallFinger", 19: "LThumb",
                       20: "RShoulder", 21: "RElbow", 22: "RWrist",
                       23: "RSmallFinger", 24: "RThumb"}
CENTER_JOINT_H36M_25 = 0
LIMBSEQ_H36M_25 = [[0, 1], [0,6], #hips
                   [1,2], [2,3], [3,4], [4,5], # right foot
                   [6,7], [7,8], [8,9], [9,10], # left foot
                   [0,11], [11,12], [12,13], [13,14], # torso and head
                   [12,15], [12,20], # shoulders
                   [15,16], [16,17], [17,18], [17,19], # left hand
                    [20,21], [21,22], [22,23], [22,24] # right hand
                   ]


# removing feet and hands results in a 17 representation, which DIFFERS from other 17-joints representations (COCO, OpenPose, FreeMan, CHeck again for COCO and OpenPose). It is the same as BelFusion.
CONVERSION_IDX_H36M_32TO17 = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]

JOINTS_DICT_H36M_17 = {0: 'Hip', 1: 'RHip', 2: 'RKnee', 3: 'RAnkle',       4: 'LHip', 5: 'LKnee', 6: 'LAnkle', 
                                 7: 'Torso', 8: 'Neck', 9: 'Nose', 10: 'Head', 
                                 11: 'LShoulder', 12: 'LElbow', 13: 'LWrist',        14: 'RShoulder', 15: 'RElbow',16: 'RWrist'}

CENTER_JOINT_H36M_17 = 0
LIMBSEQ_H36M_17 = [[0, 1], [0,4], #hips
                   [1,2], [2,3], # right foot
                   [4,5], [5,6], # left foot
                   [0,7], [7,8], [8,9], [9,10], # torso and head
                   [8,11], [8,14], # shoulders
                   [11,12], [12,13], # left hand
                    [14,15], [15,16] # right hand
                   ]


assert len(JOINTS_DICT_H36M_32) == 32
assert len(JOINTS_DICT_H36M_25) == 25
assert len(CONVERSION_IDX_H36M_32TO25) == 25
assert len(JOINTS_DICT_H36M_17) == 17
assert len(CONVERSION_IDX_H36M_32TO17) == 17


class H36MKinematic(Kinematic):
    def __init__(self, num_joints=17, **kwargs):
        
        super().__init__(**kwargs)
        
        assert num_joints in [17, 25]
        if num_joints == 17:
            self.joint_dict_orig  = JOINTS_DICT_H36M_17.copy()
            self.limbseq = LIMBSEQ_H36M_17.copy()
        elif num_joints == 25:
            self.joint_dict_orig  = JOINTS_DICT_H36M_25.copy()
            self.limbseq = LIMBSEQ_H36M_25.copy()
        self.joint_dict_orig[0] = 'GlobalRoot' 
        self.left_right_limb_list = [False if joint[0] == "L" and joint[1].isupper() else True for joint in list(self.joint_dict_orig.values())]

        #    self.node_limbseq = [[limb[0] -1,  limb[1] -1] for limb in self.limbseq if limb[1] != 0 and limb[0] != 0]
        #       # rememeber to add hip connections!
        # else: 
        #     self.node_dict =  self.joint_dict_orig
        #    self.node_limbseq = self.limbseq

        if not self.if_consider_hip:
            self.node_dict = self.joint_dict_orig.copy()
            self.node_dict.pop(0)
            self.node_dict = {i:v for i,v in enumerate(list(self.node_dict.values()))}
            node_dict_reversed = {v:i for i,v in self.node_dict.items()}
            self.node_limbseq = [[node_dict_reversed['RHip'], node_dict_reversed['LHip']], 
                                    [node_dict_reversed['RHip'], node_dict_reversed['Torso']], 
                                    [node_dict_reversed['LHip'], node_dict_reversed['Torso']],
                                    *[[limb[0] -1,  limb[1] -1] for limb in self.limbseq if limb[1] != 0 and limb[0] != 0]]
            assert num_joints == 17
            self.limb_angles_idx = [[3,4], [0,2,7,8,9], [1,7,10,12,13], [7,11,14,15]]
            self.kinchain = [[0,6,7,8,9], # hip to head 
                             [7,10,11,12], # left arm
                             [7,13,14,15], # right arm
                                [3,4,5], # left leg
                                [0,1,2], # right leg
                                [0,3,6], # hip triangle
                              ]
        else: 
            self.node_dict = {k:v for k,v in enumerate(list(self.node_hip.values())+list(self.joint_dict_orig.values())[1:])}
            self.node_limbseq = self.limbseq.copy()
            # self.limb_angles_idx = [[3,4], [0,2,7,8,9], [1,7,10,12,13], [7,11,14,15]]
        self.limbseq = np.array(self.limbseq)
