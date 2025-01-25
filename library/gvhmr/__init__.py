
import cv2
import torch
import pytorch_lightning as pl
import numpy as np
import argparse
from hmr4d.utils.pylogger import Log
import hydra
from hydra import initialize_config_module, compose
from pytorch3d.transforms import quaternion_to_matrix
from pathlib import Path

from hmr4d.configs import register_store_gvhmr
from hmr4d.utils.video_io_utils import (
    get_video_lwh,
    read_video_np,
    save_video,
    get_writer,
    get_video_reader,
)
from hmr4d.utils.vis.cv2_utils import draw_bbx_xyxy_on_image_batch, draw_coco17_skeleton_batch

from hmr4d.utils.preproc import Tracker, Extractor, VitPoseExtractor, SLAMModel

from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy, estimate_K, convert_K_to_K4, create_camera_sensor
from hmr4d.utils.geo_transform import compute_cam_angvel
from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
from hmr4d.utils.net_utils import detach_to_cpu, to_cuda
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.vis.renderer import Renderer, get_global_cameras_static, get_ground_params_from_points
from tqdm import tqdm
from hmr4d.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay
from einops import einsum, rearrange
from hmr4d.utils.bvh import bvh, quat


class GvhmrInfer:
    def __init__(self):
        self.model = None
        self.cfg_global = None
        self.offsets = None
        self.parents = None
        self.tracker_model = None
        self.vitpose_model = None
        self.vit_extractor = None

    def load_model(self):
        with initialize_config_module(version_base="1.3", config_module=f"hmr4d.configs"):
            register_store_gvhmr()
            self.cfg_global = compose(config_name="demo") # Load default config
        
        self.tracker_model = Tracker()
        self.vitpose_model = VitPoseExtractor()
        self.vit_extractor = Extractor()

        self.model: DemoPL = hydra.utils.instantiate(self.cfg_global.model, _recursive_=False)
        self.model.load_pretrained_model(self.cfg_global.ckpt_path)
        self.model = self.model.eval().cuda()
        Log.info("Model loaded successfully.")

    def load_data_dict(self, cfg):
        paths = cfg.paths
        length, width, height = get_video_lwh(cfg.video_path)
        if cfg.static_cam:
            R_w2c = torch.eye(3).repeat(length, 1, 1)
        else:
            traj = torch.load(cfg.paths.slam)
            traj_quat = torch.from_numpy(traj[:, [6, 3, 4, 5]])
            R_w2c = quaternion_to_matrix(traj_quat).mT
        K_fullimg = estimate_K(width, height).repeat(length, 1, 1)
        # K_fullimg = create_camera_sensor(width, height, 26)[2].repeat(length, 1, 1)

        data = {
            "length": torch.tensor(length),
            "bbx_xys": torch.load(paths.bbx)["bbx_xys"],
            "kp2d": torch.load(paths.vitpose),
            "K_fullimg": K_fullimg,
            "cam_angvel": compute_cam_angvel(R_w2c),
            "f_imgseq": torch.load(paths.vit_features),
        }
        return data
    def _copy_video(self, src_path, dst_path):
        """复制视频文件到工作目录"""
        reader = get_video_reader(src_path)
        writer = get_writer(dst_path, fps=30, crf=23)
        for img in tqdm(reader, desc="Copying video", total=get_video_lwh(src_path)[0]):
            writer.write_frame(img)
        writer.close()
        reader.close()

    def _run_preprocess(self, cfg):
        """执行所有预处理步骤"""
        Log.info("Running preprocessing...")
        tic = Log.time()
        
        # 边界框跟踪
        if not Path(cfg.paths.bbx).exists():
            bbx_xyxy = self.tracker_model.get_one_track(cfg.video_path).float()
            bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()
            torch.save({"bbx_xyxy": bbx_xyxy, "bbx_xys": bbx_xys}, cfg.paths.bbx)
        
        # 关键点检测
        if not Path(cfg.paths.vitpose).exists():
            vitpose = self.vitpose_model.extract(cfg.video_path, torch.load(cfg.paths.bbx)["bbx_xys"])
            torch.save(vitpose, cfg.paths.vitpose)
        
        # 特征提取
        if not Path(cfg.paths.vit_features).exists():
            features = self.vit_extractor.extract_video_features(
                cfg.video_path, 
                torch.load(cfg.paths.bbx)["bbx_xys"]
            )
            torch.save(features, cfg.paths.vit_features)
        
        # SLAM处理（动态相机）
        if not cfg.static_cam and not Path(cfg.paths.slam).exists():
            length, width, height = get_video_lwh(cfg.video_path)
            K = estimate_K(width, height)
            slam = SLAMModel(cfg.video_path, width, height, convert_K_to_K4(K), 
                            buffer=4000, resize=0.5)
            with tqdm(total=length, desc="SLAM processing") as pbar:
                while slam.track():
                    pbar.update()
            torch.save(slam.process(), cfg.paths.slam)
        
        Log.info(f"Preprocessing completed in {Log.time()-tic:.1f}s")

    @torch.no_grad()
    def infer(self, video_path, static_cam=False, verbose=False, fps=30, render=False, output_root=""):
        # 初始化配置
        video_path = Path(video_path)
        assert video_path.exists(), f"Video not found at {video_path}"
        
        # 生成Hydra配置
        with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
            register_store_gvhmr()
            overrides = [
                f"video_name={video_path.stem}",
                f"static_cam={static_cam}",
                f"verbose={verbose}",
                f"output_root={output_root}" if output_root else ""
            ]
            cfg = compose(config_name="demo", overrides=overrides)

        # 确保目录结构
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)

        # 视频预处理
        if not Path(cfg.video_path).exists() or get_video_lwh(video_path)[0] != get_video_lwh(Path(cfg.video_path))[0]:
            self._copy_video(video_path, cfg.video_path)

        # 运行预处理流程
        self._run_preprocess(cfg)

        # 加载数据
        data = self.load_data_dict(cfg)

        # 模型预测
        results_path = Path(cfg.paths.hmr4d_results)
        if not results_path.with_suffix('.pkl').exists():
            if self.model is None:
                self.load_model()
                
            # 执行预测
            pred = self.model.predict(data, static_cam=cfg.static_cam)
            pred = detach_to_cpu(pred)
            
            # 保存原始预测结果
            torch.save(pred, cfg.paths.hmr4d_results)
            
            # 准备BVH数据
            pred_np = self.prepare_bvh_data(pred)
            
            # 保存BVH文件
            self.save_bvh(pred_np, str(results_path.with_suffix('')), mirror=True, fps=fps)
            
            # 保存序列化数据
            import pickle
            with open(results_path.with_suffix('.pkl'), 'wb') as f:
                pickle.dump(pred_np, f, protocol=pickle.HIGHEST_PROTOCOL)

        # 可视化渲染
        if render:
            self._render_results(cfg)

        return str(results_path.with_suffix('.bvh'))

    def prepare_bvh_data(self, pred):
        pred_np = {}
        smpl_poses = np.concatenate([
            pred['smpl_params_global']['global_orient'].cpu().numpy(),
            pred['smpl_params_global']['body_pose'].cpu().numpy()
        ], axis=1)
        smpl_trans = pred['smpl_params_global']['transl'].cpu().numpy()
        num_frames, current_joints = smpl_poses.shape[0], smpl_poses.shape[1] // 3

        target_joints = 24
        if current_joints < target_joints:
            print(f"Current joints: {current_joints}, adding {target_joints - current_joints} missing joints...")
            additional_joints = np.zeros((num_frames, (target_joints - current_joints) * 3))
            smpl_poses = np.concatenate([smpl_poses, additional_joints], axis=1)
        # has_nan = np.isnan(smpl_poses).any()
        # all_zero_rows = np.all(smpl_poses == 0, axis=1)
        # print(f"NaN in smpl_poses: {has_nan}")
        # print(f"Rows with all zeros: {np.sum(all_zero_rows)}")
        # smpl_poses = np.nan_to_num(smpl_poses, nan=0.0, posinf=0.0, neginf=0.0)
        # if np.any(all_zero_rows):
        #     smpl_poses[all_zero_rows] = 0.001

        pred_np['smpl_poses'] = smpl_poses  # (N, 72)
        pred_np['smpl_trans'] = smpl_trans  # (N, 3)
        pred_np['smpl_scaling'] = 1.0
        return pred_np

    def mirror_rot_trans(self, lrot, trans, names, parents):
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

    def load_model_offsets(self, offsets_path: str):
        with open(offsets_path, "rb") as f:
            datas = np.load(f)
            self.offsets = datas["offsets"]
            self.parents = datas["parents"]

    def save_bvh(self, pred_np, output: str, mirror: bool,
                fps=60) -> None:
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
        names = arp_maps_bones

        if self.offsets is None or self.parents is None:
            self.load_model_offsets("inputs/model_offsets.npz")

        scaling = None

        rots = pred_np["smpl_poses"]  # (N, 72)
        rots = rots.reshape(rots.shape[0], -1, 3)  # (N, 24, 3)
        scaling = pred_np["smpl_scaling"]  # (1,)
        trans = pred_np["smpl_trans"]  # (N, 3)

        if scaling is not None:
            trans /= scaling

        # to quaternion
        rots = quat.from_axis_angle(rots)

        order = "zyx"
        pos = self.offsets[None].repeat(len(rots), axis=0)
        positions = pos.copy()

        positions[:, 0] += trans * 100
        rotations = np.degrees(quat.to_euler(rots, order=order))

        bvh_data = {
            "rotations": rotations,
            "positions": positions,
            "offsets": self.offsets,
            "parents": self.parents,
            "names": names,
            "order": order,
            "frametime": 1 / fps,
        }

        if not output.endswith(".bvh"):
            output = output + ".bvh"

        bvh.save(output, bvh_data)

        if mirror:
            rots_mirror, trans_mirror = self.mirror_rot_trans(
                    rots, trans, names, self.parents)
            positions_mirror = pos.copy()
            positions_mirror[:, 0] += trans_mirror
            rotations_mirror = np.degrees(
                quat.to_euler(rots_mirror, order=order))

            bvh_data = {
                "rotations": rotations_mirror,
                "positions": positions_mirror,
                "offsets": self.offsets,
                "parents": self.parents,
                "names": names,
                "order": order,
                "frametime": 1 / fps,
            }

            output_mirror = output.split(".")[0] + "_mirror.bvh"
            bvh.save(output_mirror, bvh_data)


if __name__ == "__main__":
    gvhmr_infer = GvhmrInfer()
    gvhmr_infer.load_model()

    result_bvh = gvhmr_infer.infer(
        video_path="/mnt/ruby/Mocap/gBR_sBM_c01_d04_mBR0_ch01.mp4",
        output_root="outputs",
        fps=60,
        render=False
    )