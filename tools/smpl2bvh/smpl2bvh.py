import torch
import numpy as np
import argparse
import pickle
import smplx

from hmr4d.utils.bvh import bvh, quat


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="inputs/checkpoints/body_models/")
    parser.add_argument("--model_type", type=str, default="smpl", choices=["smpl", "smplx"])
    parser.add_argument("--gender", type=str, default="MALE", choices=["MALE", "FEMALE", "NEUTRAL"])
    parser.add_argument("--num_betas", type=int, default=10, choices=[10, 300])
    parser.add_argument("--poses", type=str, default="data/gWA_sFM_cAll_d27_mWA5_ch20.pkl")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--output", type=str, default="data/gWA_sFM_cAll_d27_mWA5_ch20.bvh")
    parser.add_argument("--mirror", action="store_true")
    return parser.parse_args()

def mirror_rot_trans(lrot, trans, names, parents):
    joints_mirror = np.array([(
        names.index("Left"+n[5:]) if n.startswith("Right") else (
        names.index("Right"+n[4:]) if n.startswith("Left") else 
        names.index(n))) for n in names])

    mirror_pos = np.array([-1, 1, 1])
    mirror_rot = np.array([1, 1, -1, -1])
    grot = quat.fk_rot(lrot, parents)
    trans_mirror = mirror_pos * trans
    grot_mirror = mirror_rot * grot[:,joints_mirror]
    
    return quat.ik_rot(grot_mirror, parents), trans_mirror


arp_maps_bones = [
        "c_root_master.x",  # hips
        "c_thigh_fk.l",     # thigh.L
        "c_thigh_fk.r",     # thigh.R
        "c_spine_01.x",     # spine
        "c_leg_fk.l",       # shin.L
        "c_leg_fk.r",       # shin.R
        "chest",           # chest (This one doesn't have a direct match in your list, keeping original)
        "c_foot_fk.l",      # foot.L
        "c_foot_fk.r",      # foot.R
        "c_spine_02.x",    # chest.001
        "toe.L",           # toe.L (This one doesn't have a direct match, keeping original)
        "toe.R",           # toe.R (This one doesn't have a direct match, keeping original)
        "c_neck.x",         # neck
        "c_shoulder.l",     # shoulder.L
        "c_shoulder.r",     # shoulder.R
        "c_head.x",         # head
        "c_arm_fk.l",       # upper_arm.L
        "c_arm_fk.r",       # upper_arm.R
        "c_forearm_fk.l",   # forearm.L
        "c_forearm_fk.r",   # forearm.R
        "c_hand_fk.l",      # hand.L
        "c_hand_fk.r",      # hand.R
        "Left_palm",       # Left_palm (This one doesn't have a direct match, keeping original)
        "Right_palm",      # Right_palm (This one doesn't have a direct match, keeping original)
    ]


arp_bones = [
        "root_ref.x",  # hips
        "thigh_ref.l",     # thigh.L
        "thigh_ref.r",     # thigh.R
        "spine_01_ref.x",     # spine
        "leg_ref.l",       # shin.L
        "leg_ref.r",       # shin.R
        "spine_02_ref.x",           # chest (This one doesn't have a direct match in your list, keeping original)
        "foot_ref.l",      # foot.L
        "foot_ref.r",      # foot.R
        "spine_01_ref.x",    # chest.001
        "toes_ref.l",           # toe.L (This one doesn't have a direct match, keeping original)
        "toes_ref.r",           # toe.R (This one doesn't have a direct match, keeping original)
        "neck_ref.x",         # neck
        "shoulder_ref.l",     # shoulder.L
        "shoulder_ref.r",     # shoulder.R
        "head_ref.x",         # head
        "arm_ref.l",       # upper_arm.L
        "arm_ref.r",       # upper_arm.R
        "forearm_ref.l",   # forearm.L
        "forearm_ref.r",   # forearm.R
        "hand_ref.l",      # hand.L
        "hand_ref.r",      # hand.R
        "Left_palm",       # Left_palm (This one doesn't have a direct match, keeping original)
        "Right_palm",      # Right_palm (This one doesn't have a direct match, keeping original)
    ]


dollars_bones = [
        "hips",
        "thigh.R",
        "thigh.L",
        "spine",
        "shin.R",
        "shin.L",
        "chest",
        "foot.R",
        "foot.L",
        "chest",
        "toe.R",
        "toe.L",
        "neck",
        "shoulder.R",
        "shoulder.L",
        "head",
        "upper_arm.R",
        "upper_arm.L",
        "forearm.R",
        "forearm.L",
        "hand.R",
        "hand.L",
        "Right_palm",
        "Left_palm",
    ]


origin_bones = [
        "Pelvis",
        "Left_hip",
        "Right_hip",
        "Spine1",
        "Left_knee",
        "Right_knee",
        "Spine2",
        "Left_ankle",
        "Right_ankle",
        "Spine3",
        "Left_foot",
        "Right_foot",
        "Neck",
        "Left_collar",
        "Right_collar",
        "Head",
        "Left_shoulder",
        "Right_shoulder",
        "Left_elbow",
        "Right_elbow",
        "Left_wrist",
        "Right_wrist",
        "Left_palm",
        "Right_palm",
    ]


def save_model_offsets(model_path: str, model_type: str, gender: str, num_betas: int, output_path: str):
    model = smplx.create(model_path=model_path,
                        model_type=model_type,
                        gender=gender, 
                        batch_size=1)
    parents = model.parents.detach().cpu().numpy()
    rest = model()
    rest_pose = rest.joints.detach().cpu().numpy().squeeze()[:24,:]
    root_offset = rest_pose[0]
    offsets = rest_pose - rest_pose[parents]
    offsets[0] = root_offset
    offsets *= 100
    np.savez(output_path, offsets=offsets, parents=parents)


def load_model_offsets(offsets_path: str):
    with open(offsets_path, "rb") as f:
        datas = np.load(f)
        return datas["offsets"], datas["parents"]


def smpl2bvh(model_path:str, poses:str, output:str, mirror:bool,
             model_type="smpl", gender="MALE", 
             num_betas=10, fps=60) -> None:
    """Save bvh file created by smpl parameters.

    Args:
        model_path (str): Path to smpl models.
        poses (str): Path to npz or pkl file.
        output (str): Where to save bvh.
        mirror (bool): Whether save mirror motion or not.
        model_type (str, optional): I prepared "smpl" only. Defaults to "smpl".
        gender (str, optional): Gender Information. Defaults to "MALE".
        num_betas (int, optional): How many pca parameters to use in SMPL. Defaults to 10.
        fps (int, optional): Frame per second. Defaults to 30.
    """
    with open(poses, "rb") as f:
        poses = pickle.load(f)

    names = arp_maps_bones

    parents = [
        -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21
    ]
    
    save_model_offsets(model_path, model_type, gender, num_betas, "inputs/model_offsets.npz")
    offsets, parents = load_model_offsets("inputs/model_offsets.npz")
    print("Loaded Offsets:")
    print(offsets)

    scaling = None
    parents = np.array(parents)
    rots = poses["smpl_poses"] # (N, 72)
    rots = rots.reshape(rots.shape[0], -1, 3) # (N, 24, 3)
    scaling = poses["smpl_scaling"]  # (1,)
    trans = poses["smpl_trans"]  # (N, 3)
    
    if scaling is not None:
        trans /= scaling
    
    # to quaternion
    rots = quat.from_axis_angle(rots)
    
    order = "zyx"
    pos = offsets[None].repeat(len(rots), axis=0)
    positions = pos.copy()
    
    positions[:,0] += trans * 100
    rotations = np.degrees(quat.to_euler(rots, order=order))
    
    bvh_data ={
        "rotations": rotations,
        "positions": positions,
        "offsets": offsets,
        "parents": parents,
        "names": names,
        "order": order,
        "frametime": 1 / fps,
    }
    
    if not output.endswith(".bvh"):
        output = output + ".bvh"
    
    bvh.save(output, bvh_data)
    
    if mirror:
        rots_mirror, trans_mirror = mirror_rot_trans(
                rots, trans, names, parents)
        positions_mirror = pos.copy()
        positions_mirror[:,0] += trans_mirror
        rotations_mirror = np.degrees(
            quat.to_euler(rots_mirror, order=order))
        
        bvh_data ={
            "rotations": rotations_mirror,
            "positions": positions_mirror,
            "offsets": offsets,
            "parents": parents,
            "names": names,
            "order": order,
            "frametime": 1 / fps,
        }
        
        output_mirror = output.split(".")[0] + "_mirror.bvh"
        bvh.save(output_mirror, bvh_data)


if __name__ == "__main__":
    args = parse_args()
    
    smpl2bvh(model_path=args.model_path, model_type=args.model_type, 
             mirror = args.mirror, gender=args.gender,
             poses=args.poses, num_betas=args.num_betas, 
             fps=args.fps, output=args.output)
    
    print("finished!")