import numpy as np
import os
import pickle
import yaml
from typing import Any, Dict, List, Optional, Tuple
import tqdm
import io
import lmdb

import torch
import torch.nn.functional as TF_nn
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image

from vint_train.data.data_utils import (
    img_path_to_data,
    calculate_sin_cos,
    get_data_path,
    to_local_coords,
)


class ViNT_Dataset(Dataset):
    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        waypoint_spacing: int,
        min_dist_cat: int,
        max_dist_cat: int,
        min_action_distance: int,
        max_action_distance: int,
        negative_mining: bool,
        len_traj_pred: int,
        learn_angle: bool,
        context_size: int,
        context_type: str = "temporal",
        end_slack: int = 0,
        goals_per_obs: int = 1,
        normalize: bool = True,
        obs_type: str = "image",
        goal_type: str = "image",
        return_lange3d_inputs: bool = False,
        clip_model_name: str = "openai/clip-vit-base-patch16",
        gnm_mask_h: int = 60,
        gnm_mask_w: int = 80,
        **kwargs,
    ):
        """
        Main ViNT dataset class

        Args:
            data_folder (string): Directory with all the image data
            data_split_folder (string): Directory with filepaths.txt, a list of all trajectory names in the dataset split that are each seperated by a newline
            dataset_name (string): Name of the dataset [recon, go_stanford, scand, tartandrive, etc.]
            waypoint_spacing (int): Spacing between waypoints
            min_dist_cat (int): Minimum distance category to use
            max_dist_cat (int): Maximum distance category to use
            negative_mining (bool): Whether to use negative mining from the ViNG paper (Shah et al.) (https://arxiv.org/abs/2012.09812)
            len_traj_pred (int): Length of trajectory of waypoints to predict if this is an action dataset
            learn_angle (bool): Whether to learn the yaw of the robot at each predicted waypoint if this is an action dataset
            context_size (int): Number of previous observations to use as context
            context_type (str): Whether to use temporal, randomized, or randomized temporal context
            end_slack (int): Number of timesteps to ignore at the end of the trajectory
            goals_per_obs (int): Number of goals to sample per observation
            normalize (bool): Whether to normalize the distances or actions
            goal_type (str): What data type to use for the goal. The only one supported is "image" for now.
        """
        self.data_folder = data_folder
        self.data_split_folder = data_split_folder
        self.dataset_name = dataset_name
        self.kwargs = kwargs

        self.max_traj_len= self.kwargs["max_traj_len"]
        # When True, drop (traj, goal_time) pairs whose gt_costs vector is
        # "dead" (K<2 or std<1e-6 — e.g. all zeros, all sentinel 1e6).  Such
        # samples contribute zero ranking signal and are excluded from the
        # spearman/ranking metrics anyway.
        self.filter_dead_samples = bool(self.kwargs.get("filter_dead_samples", False))
        self._alive_times: Dict[str, np.ndarray] = {}

        traj_names_file = os.path.join(data_split_folder, "traj_names.txt")
        with open(traj_names_file, "r") as f:
            file_lines = f.read()
            self.traj_names = file_lines.split("\n")
        # check if traj exists, remove otherwise
        trajs_to_remove = [
            traj_name
            for traj_name in self.traj_names
            if not os.path.exists(os.path.join(data_folder, traj_name, "traj_data.pkl"))
        ]
        print(
            f"Removing {len(trajs_to_remove)} trajectories that do not have traj_data.pkl"
        )
        for traj_name in trajs_to_remove:
            self.traj_names.remove(traj_name)
        if "" in self.traj_names:
            self.traj_names.remove("")

        self.image_size = image_size
        self.waypoint_spacing = waypoint_spacing
        self.distance_categories = list(
            range(min_dist_cat, max_dist_cat + 1, self.waypoint_spacing)
        )
        self.min_dist_cat = self.distance_categories[0]
        self.max_dist_cat = self.distance_categories[-1]
        self.negative_mining = negative_mining
        if self.negative_mining:
            self.distance_categories.append(-1)
        self.len_traj_pred = len_traj_pred
        self.learn_angle = learn_angle

        self.min_action_distance = min_action_distance
        self.max_action_distance = max_action_distance

        self.context_size = context_size
        assert context_type in {
            "temporal",
            "randomized",
            "randomized_temporal",
        }, "context_type must be one of temporal, randomized, randomized_temporal"
        self.context_type = context_type
        self.end_slack = end_slack
        self.goals_per_obs = goals_per_obs
        self.normalize = normalize
        self.obs_type = obs_type
        self.goal_type = goal_type

        # ----- LangE3D extras -----
        self.return_lange3d_inputs = return_lange3d_inputs
        self.clip_model_name = clip_model_name
        self.gnm_mask_h = gnm_mask_h
        self.gnm_mask_w = gnm_mask_w
        self._clip_processor = None  # lazy

        # load data/data_config.yaml
        with open(
            os.path.join(os.path.dirname(__file__), "data_config.yaml"), "r"
        ) as f:
            all_data_config = yaml.safe_load(f)
        assert (
            self.dataset_name in all_data_config
        ), f"Dataset {self.dataset_name} not found in data_config.yaml"
        dataset_names = list(all_data_config.keys())
        dataset_names.sort()
        # use this index to retrieve the dataset name from the data_config.yaml
        self.dataset_index = dataset_names.index(self.dataset_name)
        self.data_config = all_data_config[self.dataset_name]
        self.images_subfolder = self.data_config.get("images_subfolder", "")
        self.images_nameformat = self.data_config.get("images_nameformat", "idx.jpg")
        self.trajectory_cache = {}
        self._load_index()
        self._build_caches()

        if self.learn_angle:
            self.num_action_params = 3
        else:
            self.num_action_params = 2

        if self.goal_type == "image_mask_enc" or self.obs_type == "image_mask_enc":
            from vint_train.models.object_react import dataloader

            self.topopaths = dataloader.TopoPaths(
                dims=self.kwargs["dims"],
                precomputed_filename=self.kwargs["precomputed_filename"],
                pl_perturb_ratio=self.kwargs["pl_perturb_ratio"],
                pl_perturb_type=self.kwargs["pl_perturb_type"],
                mask_crop_ratio=self.kwargs["mask_crop_ratio"],
                use_mask_grad=self.kwargs["use_mask_grad"],
            )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_image_cache"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._build_caches()

    def _build_caches(self, use_tqdm: bool = True):
        """
        Build a cache of images for faster loading using LMDB
        """
        cache_filename = os.path.join(
            self.data_split_folder,
            f"dataset_{self.dataset_name}.lmdb",
        )

        # Load all the trajectories into memory. These should already be loaded, but just in case.
        for traj_name in self.traj_names:
            self._get_trajectory(traj_name)

        """
        If the cache file doesn't exist, create it by iterating through the dataset and writing each image to the cache
        """
        if not os.path.exists(cache_filename):
            tqdm_iterator = tqdm.tqdm(
                self.goals_index,
                disable=not use_tqdm,
                dynamic_ncols=True,
                desc=f"Building LMDB cache for {self.dataset_name}",
            )
            with lmdb.open(cache_filename, map_size=2**40) as image_cache:
                with image_cache.begin(write=True) as txn:
                    for traj_name, time in tqdm_iterator:
                        image_path = get_data_path(
                            self.data_folder,
                            traj_name,
                            time,
                            self.images_subfolder,
                            self.images_nameformat,
                        )
                        with open(image_path, "rb") as f:
                            txn.put(image_path.encode(), f.read())

        # Reopen the cache file in read-only mode
        self._image_cache: lmdb.Environment = lmdb.open(cache_filename, readonly=True)

    def _build_index(self, use_tqdm: bool = False ):
        """
        Build an index consisting of tuples (trajectory name, time, max goal distance)
        """
        samples_index = []
        goals_index = []
        skipped = 0

        n_goals_alive = 0
        n_goals_total = 0

        for traj_name in tqdm.tqdm(
            self.traj_names, disable=not use_tqdm, dynamic_ncols=True
        ):
            traj_data = self._get_trajectory(traj_name)
            traj_len = len(traj_data["position"])

            if self.max_traj_len is not None and traj_len > self.max_traj_len:
                skipped += 1
                continue

            if self.filter_dead_samples:
                alive = self._compute_alive_times(traj_name)
            else:
                alive = None

            for goal_time in range(0, traj_len):
                n_goals_total += 1
                if alive is not None and not (goal_time < alive.shape[0] and alive[goal_time]):
                    continue
                n_goals_alive += 1
                goals_index.append((traj_name, goal_time))

            begin_time = self.context_size * self.waypoint_spacing
            end_time = (
                traj_len - self.end_slack - self.len_traj_pred * self.waypoint_spacing
            )
            for curr_time in range(begin_time, end_time):
                if alive is not None and not (curr_time < alive.shape[0] and alive[curr_time]):
                    continue
                max_goal_distance = min(
                    self.max_dist_cat * self.waypoint_spacing, traj_len - curr_time - 1
                )
                samples_index.append((traj_name, curr_time, max_goal_distance))
        #filtering samples with repeated steps        
        if self.max_traj_len is not None:
            print(f"Skipped {skipped} trajectories longer than {self.max_traj_len} steps")
        if self.filter_dead_samples and n_goals_total > 0:
            kept = n_goals_alive / n_goals_total
            print(
                f"filter_dead_samples=True: kept {n_goals_alive}/{n_goals_total} "
                f"goal frames ({100*kept:.1f}% alive); "
                f"dropped {n_goals_total - n_goals_alive} dead frames "
                f"(K<2 or std<=1e-6)"
            )
        return samples_index, goals_index

    def _sample_goal(self, trajectory_name, curr_time, max_goal_dist):
        """
        Sample a goal from the future in the same trajectory.
        Returns: (trajectory_name, goal_time, goal_is_negative)
        """
        goal_offset = np.random.randint(0, max_goal_dist + 1)
        if goal_offset == 0:
            trajectory_name, goal_time = self._sample_negative()
            return trajectory_name, goal_time, True
        else:
            goal_time = curr_time + int(goal_offset * self.waypoint_spacing)
            if self.filter_dead_samples:
                # Try a few re-samples within the future window to land on an
                # alive frame; otherwise fall back to a negative.
                for _ in range(8):
                    if self._is_alive(trajectory_name, goal_time):
                        return trajectory_name, goal_time, False
                    off = np.random.randint(1, max_goal_dist + 1)
                    goal_time = curr_time + int(off * self.waypoint_spacing)
                if not self._is_alive(trajectory_name, goal_time):
                    trajectory_name, goal_time = self._sample_negative()
                    return trajectory_name, goal_time, True
            return trajectory_name, goal_time, False

    def _sample_negative(self):
        """
        Sample a goal from a (likely) different trajectory.
        """
        return self.goals_index[np.random.randint(0, len(self.goals_index))]

    def _compute_alive_times(self, traj_name: str) -> np.ndarray:
        """Return per-timestep bool array marking frames with usable gt_costs.

        A frame is "alive" if its gt_costs vector has K>=2 objects AND
        std(gt_costs) > 1e-6 (i.e. not all-zeros and not all-sentinel).
        Frames without gt_costs are marked dead.
        """
        if traj_name in self._alive_times:
            return self._alive_times[traj_name]
        td = self._get_trajectory(traj_name)
        T = len(td["position"]) if isinstance(td, dict) and "position" in td else 0
        alive = np.zeros(T, dtype=bool)
        if isinstance(td, dict) and "gt_costs" in td:
            gtc = td["gt_costs"]
            for t in range(T):
                try:
                    arr = np.asarray(gtc[t], dtype=np.float32) if t < len(gtc) else None
                except Exception:
                    arr = None
                if arr is None or arr.size < 2:
                    continue
                if float(arr.std()) > 1e-6:
                    alive[t] = True
        self._alive_times[traj_name] = alive
        return alive

    def _is_alive(self, traj_name: str, t: int) -> bool:
        alive = self._compute_alive_times(traj_name)
        return 0 <= t < alive.shape[0] and bool(alive[t])

    def _load_index(self) -> None:
        """
        Generates a list of tuples of (obs_traj_name, goal_traj_name, obs_time, goal_time) for each observation in the dataset
        """
        suffix_maxlen = (
            f"_maxlen{self.max_traj_len}" if self.max_traj_len is not None else ""
        )
        suffix_alive = "_aliveonly" if self.filter_dead_samples else ""
        index_to_data_path = os.path.join(
            self.data_split_folder,
            f"dataset_dist_{self.min_dist_cat}_to_{self.max_dist_cat}"
            f"_context_{self.context_type}_n{self.context_size}"
            f"_slack_{self.end_slack}{suffix_maxlen}{suffix_alive}.pkl",
        )
        try:
            # load the index_to_data if it already exists (to save time)
            with open(index_to_data_path, "rb") as f:
                self.index_to_data, self.goals_index = pickle.load(f)
        except:
            # if the index_to_data file doesn't exist, create it
            self.index_to_data, self.goals_index = self._build_index()
            with open(index_to_data_path, "wb") as f:
                pickle.dump((self.index_to_data, self.goals_index), f)

    def _load_image(self, trajectory_name, time):
        image_path = get_data_path(
            self.data_folder,
            trajectory_name,
            time,
            self.images_subfolder,
            self.images_nameformat,
        )

        try:
            with self._image_cache.begin() as txn:
                image_buffer = txn.get(image_path.encode())
                image_bytes = bytes(image_buffer)
            image_bytes = io.BytesIO(image_bytes)
            return img_path_to_data(image_bytes, self.image_size)
        except (TypeError, OSError, ValueError) as e:
            print(f"Failed to load image {image_path}: {e}")

    def _compute_actions(self, traj_data, curr_time, goal_time):
        start_index = curr_time
        end_index = curr_time + self.len_traj_pred * self.waypoint_spacing + 1
        yaw = traj_data["yaw"][start_index : end_index : self.waypoint_spacing]
        positions = traj_data["position"][
            start_index : end_index : self.waypoint_spacing
        ]
        goal_pos = traj_data["position"][min(goal_time, len(traj_data["position"]) - 1)]

        if len(yaw.shape) == 2:
            yaw = yaw.squeeze(1)

        if yaw.shape != (self.len_traj_pred + 1,):
            const_len = self.len_traj_pred + 1 - yaw.shape[0]
            yaw = np.concatenate([yaw, np.repeat(yaw[-1], const_len)])
            positions = np.concatenate(
                [positions, np.repeat(positions[-1][None], const_len, axis=0)], axis=0
            )

        assert yaw.shape == (
            self.len_traj_pred + 1,
        ), f"{yaw.shape} and {(self.len_traj_pred + 1,)} should be equal"
        assert positions.shape == (
            self.len_traj_pred + 1,
            2,
        ), f"{positions.shape} and {(self.len_traj_pred + 1, 2)} should be equal"

        waypoints = to_local_coords(positions, positions[0], yaw[0])
        goal_pos = to_local_coords(goal_pos, positions[0], yaw[0])

        assert waypoints.shape == (
            self.len_traj_pred + 1,
            2,
        ), f"{waypoints.shape} and {(self.len_traj_pred + 1, 2)} should be equal"

        if self.learn_angle:
            yaw = yaw[1:] - yaw[0]
            actions = np.concatenate([waypoints[1:], yaw[:, None]], axis=-1)
        else:
            actions = waypoints[1:]

        if self.normalize:
            actions[:, :2] /= (
                self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing
            )
            goal_pos /= (
                self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing
            )

        assert actions.shape == (
            self.len_traj_pred,
            self.num_action_params,
        ), f"{actions.shape} and {(self.len_traj_pred, self.num_action_params)} should be equal"

        return actions, goal_pos

    def _get_trajectory(self, trajectory_name):
        if trajectory_name in self.trajectory_cache:
            return self.trajectory_cache[trajectory_name]
        else:
            with open(
                os.path.join(self.data_folder, trajectory_name, "traj_data.pkl"), "rb"
            ) as f:
                traj_data = pickle.load(f)
            # Only cast ndarray values; keep strings/lists (e.g. "instruction",
            # "gt_costs", "cat_names") as-is so the LangE3D path can use them.
            traj_data = {
                k: (v.astype(float) if isinstance(v, np.ndarray) else v)
                for k, v in traj_data.items()
            }
            self.trajectory_cache[trajectory_name] = traj_data
            return traj_data

    # ------------------------------------------------------------------
    # LangE3D goal-frame inputs (raw RGB / masks / pls / instruction)
    # ------------------------------------------------------------------
    def _get_clip_processor(self):
        if self._clip_processor is None:
            from transformers import CLIPProcessor
            self._clip_processor = CLIPProcessor.from_pretrained(
                self.clip_model_name
            )
        return self._clip_processor

    def _load_lange3d_goal_inputs(self, f_goal: str, goal_time: int) -> Dict[str, Any]:
        """
        Returns the inputs needed to (a) run ``LangGeoNetV2`` on the goal
        frame and (b) compute its supervised loss.

        Keys:
            pixel_values_goal  : [3, 224, 224] CLIP-preprocessed RGB
            masks_goal         : [K, H, W] bool — full-res per-instance masks
            gnm_masks          : [K, gnm_mask_h, gnm_mask_w] float32 — masks
                                 resized for ``TopoPaths.build_differentiable_goal``
            gt_costs           : [K] float32 — raw path-length costs (targets)
            nai_input_ids      : [77] CLIP text tokens (episode instruction)
            nai_attention_mask : [77]
            K                  : int
        """
        # ---- Goal RGB → CLIP processor (native resolution) -----------
        rgb_path = get_data_path(
            self.data_folder, f_goal, goal_time,
            self.images_subfolder, self.images_nameformat,
        )
        pil = Image.open(rgb_path).convert("RGB")
        proc = self._get_clip_processor()
        pixel_values_goal = proc(
            images=pil, return_tensors="pt",
        )["pixel_values"].squeeze(0)              # [3, 224, 224]

        masks_np = np.zeros((0, 1, 1), dtype=np.uint8)
        gt_costs = np.zeros((0,), dtype=np.float32)

        precomputed = self.kwargs.get("precomputed_filename", None)
        if precomputed is not None:
            # ---- Masks + gt_costs from the precomputed H5 ------------
            import h5py
            from vint_train.models.object_react.dataloader import rle_to_mask
            key = f"{f_goal}_{goal_time}"
            with h5py.File(precomputed, "r") as f:
                if key in f:
                    kd = f[key]
                    img_size = kd["size"][()]
                    img_masks_grp = kd["img_masks"]
                    K = len(img_masks_grp.keys())
                    rles = [
                        {"size": img_size, "counts": img_masks_grp[f"{mi}"][()]}
                        for mi in range(K)
                    ]
                    if K > 0:
                        masks_np = np.stack(
                            [rle_to_mask(r) for r in rles], axis=0
                        ).astype(np.uint8)          # [K, H, W]
                    gt_costs = np.asarray(
                        kd["img_pls"][()], dtype=np.float32
                    )
        else:
            # ---- Fall back: masks/*.npz + traj_data.pkl (convert_h5_to_vint layout)
            npz_path = os.path.join(
                self.data_folder, f_goal, "masks", f"{goal_time}.npz",
            )
            if os.path.isfile(npz_path):
                masks_np = np.load(npz_path)["masks"].astype(np.uint8)  # [K, H, W]
            goal_td = self._get_trajectory(f_goal)
            if (
                isinstance(goal_td, dict)
                and "gt_costs" in goal_td
                and goal_time < len(goal_td["gt_costs"])
            ):
                gt_costs = np.asarray(
                    goal_td["gt_costs"][goal_time], dtype=np.float32,
                )
                # Align K between masks and costs
                K_m = masks_np.shape[0]
                K_c = gt_costs.shape[0]
                if K_m != K_c:
                    K_eff = min(K_m, K_c)
                    masks_np = masks_np[:K_eff]
                    gt_costs = gt_costs[:K_eff]

        K = masks_np.shape[0]
        # Resize masks to GNM goal-encoder grid for build_differentiable_goal
        if K == 0:
            gnm_masks = np.zeros(
                (0, self.gnm_mask_h, self.gnm_mask_w), dtype=np.float32
            )
        else:
            t = torch.from_numpy(masks_np).unsqueeze(0).float()
            t = TF_nn.interpolate(
                t, size=(self.gnm_mask_h, self.gnm_mask_w), mode="nearest",
            )
            gnm_masks = t.squeeze(0).numpy()

        # ---- Instruction → CLIP text tokens --------------------------
        goal_td = self._get_trajectory(f_goal)
        instruction = goal_td.get("instruction", "") if isinstance(goal_td, dict) else ""
        if not isinstance(instruction, str):
            try:
                instruction = instruction.decode("utf-8")
            except Exception:
                instruction = str(instruction)
        if not instruction:
            instruction = "navigate"
        nai = proc(
            text=instruction,
            padding="max_length", truncation=True,
            max_length=77, return_tensors="pt",
        )

        return {
            "pixel_values_goal":  pixel_values_goal,
            "masks_goal":         torch.from_numpy(masks_np).bool(),
            "gnm_masks":          torch.from_numpy(gnm_masks),
            "gt_costs":           torch.from_numpy(gt_costs),
            "nai_input_ids":      nai["input_ids"].squeeze(0),
            "nai_attention_mask": nai["attention_mask"].squeeze(0),
            "instruction":        instruction,
            "K":                  K,
        }

    def __len__(self) -> int:
        return len(self.index_to_data)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        """
        Args:
            i (int): index to ith datapoint
        Returns:
            Tuple of tensors containing the context, observation, goal, transformed context, transformed observation, transformed goal, distance label, and action label
                obs_image (torch.Tensor): tensor of shape [3, H, W] containing the image of the robot's observation
                goal_image (torch.Tensor): tensor of shape [3, H, W] containing the subgoal image
                dist_label (torch.Tensor): tensor of shape (1,) containing the distance labels from the observation to the goal
                action_label (torch.Tensor): tensor of shape (5, 2) or (5, 4) (if training with angle) containing the action labels from the observation to the goal
                which_dataset (torch.Tensor): index of the datapoint in the dataset [for identifying the dataset for visualization when using multiple datasets]
        """
        f_curr, curr_time, max_goal_dist = self.index_to_data[i]
        f_goal, goal_time, goal_is_negative = self._sample_goal(
            f_curr, curr_time, max_goal_dist
        )

        # Load images
        context = []
        if self.context_type == "temporal":
            # sample the last self.context_size times from interval [0, curr_time)
            context_times = list(
                range(
                    curr_time + -self.context_size * self.waypoint_spacing,
                    curr_time + 1,
                    self.waypoint_spacing,
                )
            )
            context = [(f_curr, t) for t in context_times]
        else:
            raise ValueError(f"Invalid context type {self.context_type}")

        if self.obs_type == "image_mask_enc":
            obs_img, obs_vis = [], []
            for f, t in context:
                oimg, ovis = self.topopaths.get_topo_path(f, t, getFt=True)
                obs_img.append(oimg)
                obs_vis.append(ovis)
            obs_image = torch.as_tensor(
                np.concatenate([obs_vis[-1], np.concatenate(obs_img, 0)], 0),
                dtype=torch.float32,
            )
        elif self.obs_type == "image":
            obs_image = torch.cat([self._load_image(f, t) for f, t in context])
        elif self.obs_type == "disabled":
            obs_image = self._load_image(f_curr, curr_time)
        else:
            raise ValueError(f"Invalid observation type {self.obs_type}")

        # Load goal image
        if self.goal_type == "image_mask_enc":
            if self.return_lange3d_inputs:
                # Goal will be replaced at runtime by LangGeoNetV2 prediction;
                # return a zero placeholder of the correct shape [3+dims, mh, mw].
                dims = self.kwargs.get("dims", 8)
                mh = self.kwargs.get("gnm_mask_h", 60)
                mw = self.kwargs.get("gnm_mask_w", 80)
                goal_image = np.zeros((3 + dims, mh, mw), dtype=np.float32)
            else:
                if self.kwargs["goal_uses_context"]:
                    goal_context = context
                else:
                    goal_context = [(f_curr, curr_time)]
                goal_image_list, goal_vis_list = [], []
                for f, t in goal_context:
                    goal_image, goal_vis = self.topopaths.get_topo_path(f, t)
                    goal_image_list.append(goal_image)
                    goal_vis_list.append(goal_vis)
                goal_image = np.concatenate(goal_image_list, 0)
                goal_image = np.concatenate([goal_vis_list[-1], goal_image], axis=0)
                if goal_image.dtype == object:
                    # print("Error: goal_image is object")
                    # TODO: remove hard coded shape
                    goal_image = np.concatenate(
                        [
                            np.zeros((3, 60, 80)),
                            np.ones((self.kwargs["dims_segFt"], 60, 80)),
                        ],
                        axis=0,
                    )
        elif self.goal_type == "image":
            goal_image = self._load_image(f_goal, goal_time)
        elif self.goal_type == "disabled":
            goal_image = torch.zeros(obs_image.shape[1:], dtype=torch.float32)

        # Load other trajectory data
        curr_traj_data = self._get_trajectory(f_curr)
        curr_traj_len = len(curr_traj_data["position"])
        assert curr_time < curr_traj_len, f"{curr_time} and {curr_traj_len}"

        goal_traj_data = self._get_trajectory(f_goal)
        goal_traj_len = len(goal_traj_data["position"])
        assert goal_time < goal_traj_len, f"{goal_time} an {goal_traj_len}"

        # Compute actions
        actions, goal_pos = self._compute_actions(curr_traj_data, curr_time, goal_time)

        # Compute distances
        if goal_is_negative:
            distance = self.max_dist_cat
        else:
            distance = (goal_time - curr_time) // self.waypoint_spacing
            assert (
                goal_time - curr_time
            ) % self.waypoint_spacing == 0, f"{goal_time} and {curr_time} should be separated by an integer multiple of {self.waypoint_spacing}"

        actions_torch = torch.as_tensor(actions, dtype=torch.float32)
        if self.learn_angle:
            actions_torch = calculate_sin_cos(actions_torch)

        action_mask = (
            (distance < self.max_action_distance)
            and (distance > self.min_action_distance)
            and (not goal_is_negative)
        )

        out = (
            torch.as_tensor(obs_image, dtype=torch.float32),
            torch.as_tensor(goal_image, dtype=torch.float32),
            actions_torch,
            torch.as_tensor(distance, dtype=torch.int64),
            torch.as_tensor(goal_pos, dtype=torch.float32),
            torch.as_tensor(self.dataset_index, dtype=torch.int64),
            torch.as_tensor(action_mask, dtype=torch.float32),
        )
        if self.return_lange3d_inputs:
            out = out + (self._load_lange3d_goal_inputs(f_curr, curr_time),) # goal_time
        return out
