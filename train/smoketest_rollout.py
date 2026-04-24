"""
smoketest_rollout.py
====================
Closed-loop-style rollout evaluation:

  For each episode:
    1. Train the joint GNM + LangGeoNetV2 stack on a small set of
       (curr_time, goal_time=end-of-episode) pairs that span the
       action-mask window, so the model sees a variety of starting
       positions within the same episode.
    2. Perform a **rollout from the first frame** (or the earliest frame
       with valid context) toward the episode's final frame:
         - At each iteration, pick the real episode frame whose global
           position is closest to the simulated agent position.
         - Run the model with goal_time = T-1.
         - Take the first predicted waypoint (local frame), denormalise,
           rotate by current yaw, add to current position → new pose.
         - Repeat until the simulated position is within a threshold of
           the goal position, or we exhaust `max_steps`.

  Report, per episode:
     • reached_goal (bool)                 
     • steps_taken / max_steps
     • final distance to goal (meters)
     • cumulative predicted path
"""
from __future__ import annotations
import argparse, os, sys, json, time, types, random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from vint_train.data.vint_dataset import ViNT_Dataset
from vint_train.models.gnm.gnm import GNM
from vint_train.models.object_react.dataloader import TopoPaths
from vint_train.training.train_utils import (
    _compute_losses, get_obs_image, get_goal_image,
    _maybe_predict_goal_with_lange3d,
)
from lange3dnet_train.model import LangGeoNetV2
from lange3dnet_train.losses import LangGeoNetLoss
from train import _collate_with_lange3d


class ListDataset(Dataset):
    def __init__(self, samples): self.samples = samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


def _set_goal(base_ds, f_goal, goal_time):
    """Monkey-patch _sample_goal to always return (f_goal, goal_time)."""
    def _const_goal(self, trajectory_name, curr_t, max_d,
                    _f=f_goal, _g=goal_time):
        return _f, _g, False
    base_ds._sample_goal = types.MethodType(_const_goal, base_ds)


def build_episode_training_samples(base_ds, n_episodes, *,
                                   len_traj_pred, min_ad, max_ad,
                                   context_size, spacing, per_ep=5):
    """For each of n_episodes trajectories, collect up to `per_ep` samples
    at different curr_times, all with goal_time = T-1 (episode end).
    All selected samples will have action_mask=1."""
    out, eps_info = [], []
    for traj_name in base_ds.traj_names:
        if len(eps_info) >= n_episodes: break
        if not traj_name: continue
        traj = base_ds._get_trajectory(traj_name)
        T = len(traj["position"])
        goal_time = T - 1
        valid_dists = list(range(min_ad + 1, max_ad))
        chosen = []
        _set_goal(base_ds, traj_name, goal_time)
        for d in valid_dists:
            curr_time = goal_time - d * spacing
            if curr_time < context_size * spacing: continue
            if curr_time + len_traj_pred * spacing + 1 > T: continue
            base_ds.index_to_data.append((traj_name, curr_time,
                                          T - curr_time - 1))
            s = base_ds[len(base_ds.index_to_data) - 1]
            if float(s[6]) > 0.5:
                chosen.append(s)
                if len(chosen) >= per_ep: break
        if len(chosen) < 2:
            continue  # skip short episodes
        out.extend(chosen)
        eps_info.append({"traj": traj_name, "T": T, "goal_time": goal_time,
                         "n_samples": len(chosen)})
        print(f"  [{len(eps_info):2d}/{n_episodes}] traj={traj_name:>6} "
              f"T={T:3d} goal={goal_time}  samples={len(chosen)}")
    return out, eps_info


def build_rollout_sample(base_ds, traj_name, curr_time, goal_time):
    """Build a single ViNT sample dict for (traj, curr, goal)."""
    _set_goal(base_ds, traj_name, goal_time)
    T = len(base_ds._get_trajectory(traj_name)["position"])
    base_ds.index_to_data.append((traj_name, curr_time, T - curr_time - 1))
    return base_ds[len(base_ds.index_to_data) - 1]


@torch.no_grad()
def predict_first_waypoint(gnm, lange3d, topopaths, sample, *,
                            obs_type, goal_type_cfg, transform, device,
                            metric_spacing, normalize):
    """Return the first predicted waypoint *in local frame, denormalised
    to metres*, along with the predicted yaw delta."""
    # Batch the sample
    has_extra = len(sample) == 8
    batch = [sample]
    data = _collate_with_lange3d(batch) if has_extra else None
    if data is None:  # fallback
        data = tuple(s.unsqueeze(0) if isinstance(s, torch.Tensor) else s
                     for s in sample)
    (obs_image, goal_image, action_label, dist_label, goal_pos,
     dataset_index, action_mask) = data[:7]
    obs_image, _ = get_obs_image(obs_image, obs_type, transform, device)
    goal_image, replaced, _ = _maybe_predict_goal_with_lange3d(
        data, goal_image, device, lange3d, topopaths,
    )
    eff_goal = "image_mask_enc" if replaced else goal_type_cfg
    goal_image, _ = get_goal_image(goal_image, eff_goal, transform, device, obs_image)
    _, action_pred = gnm(obs_image, goal_image)   # [1, L, 4]
    ap = action_pred[0].cpu().numpy()             # [L, 4]
    wp0 = ap[0, :2].copy()                        # local, NORMALISED
    if normalize:
        wp0 *= metric_spacing                     # → metres
    # yaw from cos,sin at step 0
    c, s = float(ap[0, 2]), float(ap[0, 3])
    dyaw = float(np.arctan2(s, c))
    full_pred = ap[:, :2].copy()
    if normalize:
        full_pred *= metric_spacing
    return wp0, dyaw, full_pred, ap


def rollout_episode(base_ds, gnm, lange3d, topopaths, *,
                    traj_name, goal_time, start_time, waypoint_spacing,
                    obs_type, goal_type_cfg, transform, device,
                    metric_spacing, normalize, max_ad, context_size,
                    goal_threshold=1.0, max_steps=50):
    """Closed-loop rollout of an episode. Returns a dict with the results."""
    traj = base_ds._get_trajectory(traj_name)
    positions = np.asarray(traj["position"])        # [T, 2]
    yaws      = np.asarray(traj["yaw"])             # [T]
    T = len(positions)
    goal_pos = positions[goal_time]

    # Simulated agent state starts at real start frame
    sim_pos = positions[start_time].copy()
    sim_yaw = float(yaws[start_time])
    path_sim = [sim_pos.copy()]
    step_logs = []

    # Iterate ONE REAL FRAME AT A TIME, starting from start_time, until we
    # reach the goal_time (or `max_steps`, whichever comes first).  This is
    # exactly the procedure the user asked for:
    #   "start from the first frame, travel along the first predicted
    #    waypoint, then go to the next frame, repeat..."
    curr_time = start_time
    for step in range(max_steps):
        if curr_time >= goal_time:
            break

        # 2. Build sample + predict first waypoint
        sample = build_rollout_sample(base_ds, traj_name, curr_time, goal_time)
        real_yaw = float(yaws[curr_time])
        wp0_local, dyaw_pred, full_pred_local, raw_ap = predict_first_waypoint(
            gnm, lange3d, topopaths, sample,
            obs_type=obs_type, goal_type_cfg=goal_type_cfg,
            transform=transform, device=device,
            metric_spacing=metric_spacing, normalize=normalize,
        )

        # 3. Rotate local waypoint by CURRENT FRAME yaw (that's the frame the
        #    model thought it was at) and add to CURRENT FRAME position.
        #    (Labels are built in local coords w.r.t. positions[curr_time],
        #    yaws[curr_time] — see _compute_actions in vint_dataset.py.)
        rot = np.array([[np.cos(real_yaw), -np.sin(real_yaw)],
                        [np.sin(real_yaw),  np.cos(real_yaw)]])
        step_vec_global = rot @ wp0_local
        new_sim_pos = positions[curr_time] + step_vec_global
        new_sim_yaw = real_yaw + dyaw_pred

        dist_to_goal = float(np.linalg.norm(new_sim_pos - goal_pos))
        step_logs.append({
            "step": step, "curr_time": int(curr_time),
            "sim_pos_before": sim_pos.tolist(),
            "real_pos": positions[curr_time].tolist(),
            "wp0_local": wp0_local.tolist(),
            "new_sim_pos": new_sim_pos.tolist(),
            "dist_to_goal": dist_to_goal,
        })

        sim_pos = new_sim_pos
        sim_yaw = new_sim_yaw
        path_sim.append(sim_pos.copy())

        # Advance to the NEXT REAL FRAME (this is the "go to the next
        # frame" in the user's procedure).
        curr_time += 1

        if dist_to_goal < goal_threshold:
            return {
                "traj": traj_name, "reached": True,
                "steps_taken": step + 1, "max_steps": max_steps,
                "final_dist": dist_to_goal,
                "final_pos": sim_pos.tolist(),
                "goal_pos": goal_pos.tolist(),
                "path_sim": [p.tolist() for p in path_sim],
                "gt_path": positions[start_time:goal_time + 1].tolist(),
                "steps": step_logs,
            }

    return {
        "traj": traj_name, "reached": False,
        "steps_taken": max_steps, "max_steps": max_steps,
        "final_dist": float(np.linalg.norm(sim_pos - goal_pos)),
        "final_pos": sim_pos.tolist(),
        "goal_pos": goal_pos.tolist(),
        "path_sim": [p.tolist() for p in path_sim],
        "gt_path": positions[start_time:goal_time + 1].tolist(),
        "steps": step_logs,
    }


def plot_episode(result, save_path):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    gt = np.asarray(result["gt_path"])
    sim = np.asarray(result["path_sim"])
    gp = np.asarray(result["goal_pos"])
    ax.plot(gt[:, 0], gt[:, 1], "-", color="magenta", label="GT episode path")
    ax.plot(sim[:, 0], sim[:, 1], "-o", color="cyan", label="rollout", markersize=3)
    ax.plot(gt[0, 0], gt[0, 1], "go", label="start", markersize=10)
    ax.plot(gp[0],   gp[1],    "r*", label="goal",  markersize=15)
    ax.set_aspect("equal", "box")
    ax.legend()
    ax.set_title(f"{result['traj']} "
                 f"reached={result['reached']}  "
                 f"final_dist={result['final_dist']:.2f}m  "
                 f"steps={result['steps_taken']}")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def main():
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n_episodes", type=int, default=10)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lange3d_lr", type=float, default=5e-5)
    ap.add_argument("--lambda_lange3d", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--per_ep", type=int, default=5)
    ap.add_argument("--clip_grad", type=float, default=1.0)
    ap.add_argument("--goal_threshold_m", type=float, default=1.0)
    ap.add_argument("--max_rollout_steps", type=int, default=30)
    ap.add_argument("--out_dir", default="./smoketest_rollout")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open("config/defaults.yaml") as f: cfg = yaml.safe_load(f)
    with open(args.config) as f: cfg.update(yaml.safe_load(f))

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    kwargs = dict(
        predict_dists=cfg.get("predict_dists", True),
        precomputed_filename=cfg.get("precomputed_filename", None),
        pl_perturb_ratio=cfg.get("pl_perturb_ratio", 0.0),
        pl_perturb_type=cfg.get("pl_perturb_type", "max_val"),
        mask_crop_ratio=cfg.get("mask_crop_ratio", 1.0),
        use_mask_grad=cfg.get("use_mask_grad", False),
        goal_type=cfg["goal_type"], obs_type=cfg["obs_type"],
        dims=cfg["dims"], goal_uses_context=cfg.get("goal_uses_context", False),
        return_lange3d_inputs=bool(cfg.get("use_lange3d", False)),
        clip_model_name=cfg.get("lange3d_clip_model", "openai/clip-vit-base-patch16"),
        gnm_mask_h=cfg.get("gnm_mask_h", 60), gnm_mask_w=cfg.get("gnm_mask_w", 80),
    )
    ds_cfg = cfg["datasets"]["object_react"]
    spacing  = ds_cfg["waypoint_spacing"]
    min_ad, max_ad = cfg["action"]["min_dist_cat"], cfg["action"]["max_dist_cat"]

    print("Building ViNT_Dataset…")
    base_ds = ViNT_Dataset(
        data_folder=ds_cfg["data_folder"], data_split_folder=ds_cfg["train"],
        dataset_name="object_react", image_size=cfg["image_size"],
        waypoint_spacing=spacing,
        min_dist_cat=cfg["distance"]["min_dist_cat"],
        max_dist_cat=cfg["distance"]["max_dist_cat"],
        min_action_distance=min_ad, max_action_distance=max_ad,
        negative_mining=False,
        len_traj_pred=cfg["len_traj_pred"], learn_angle=cfg["learn_angle"],
        context_size=cfg["context_size"], context_type=cfg["context_type"],
        end_slack=ds_cfg["end_slack"], goals_per_obs=1, normalize=cfg["normalize"],
        **kwargs,
    )
    metric_spacing = base_ds.data_config["metric_waypoint_spacing"] * spacing
    print(f"metric_waypoint_spacing * spacing = {metric_spacing} m/unit")

    print(f"\nCollecting training samples: {args.n_episodes} episodes, "
          f"up to {args.per_ep} (curr_time) per episode, goal=last frame…")
    samples, eps_info = build_episode_training_samples(
        base_ds, args.n_episodes,
        len_traj_pred=cfg["len_traj_pred"], min_ad=min_ad, max_ad=max_ad,
        context_size=cfg["context_size"], spacing=spacing, per_ep=args.per_ep,
    )
    print(f"Total training samples = {len(samples)}  across "
          f"{len(eps_info)} episodes")

    loader = DataLoader(
        ListDataset(samples), batch_size=min(args.batch_size, len(samples)),
        shuffle=True,
        collate_fn=_collate_with_lange3d if kwargs["return_lange3d_inputs"] else None,
    )

    print("Building GNM + LangGeoNetV2…")
    gnm = GNM(
        cfg["context_size"], cfg["len_traj_pred"], cfg["learn_angle"],
        cfg["obs_encoding_size"], cfg["goal_encoding_size"], **kwargs,
    ).to(device)
    lange3d = topopaths = lange3d_loss_fn = None
    if cfg.get("use_lange3d", False):
        lange3d = LangGeoNetV2(
            d_model=cfg.get("lange3d_d_model", 256),
            n_heads=cfg.get("lange3d_n_heads", 8),
            n_layers=cfg.get("lange3d_n_layers", 2),
            clip_model_name=cfg.get("lange3d_clip_model", "openai/clip-vit-base-patch16"),
            dino_model_name=cfg.get("lange3d_dino_model", "facebook/dinov2-small"),
            freeze_clip=cfg.get("lange3d_freeze_clip", True),
            freeze_dino=cfg.get("lange3d_freeze_dino", True),
        ).to(device)
        topopaths = TopoPaths(
            dims=cfg["dims"], w=cfg.get("mask_w", 160), h=cfg.get("mask_h", 120),
        )
        lange3d_loss_fn = LangGeoNetLoss(
            lambda_rank=float(cfg.get("lange3d_lambda_rank", 1.0)),
            lambda_si=float(cfg.get("lange3d_lambda_si", 1.0)),
        ).to(device)

    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, gnm.parameters()),
                            lr=args.lr)
    if lange3d is not None:
        lp = [p for p in lange3d.parameters() if p.requires_grad]
        if lp: opt.add_param_group({"params": lp, "lr": args.lange3d_lr})

    lam = float(args.lambda_lange3d)
    print(f"\nTraining: lambda_lange3d={lam}  clip_grad={args.clip_grad}  "
          f"steps={args.steps}")
    gnm.train();  lange3d.train() if lange3d is not None else None
    t0 = time.time(); step = 0
    while step < args.steps:
        for data in loader:
            (obs_image, goal_image, action_label, dist_label, goal_pos,
             dataset_index, action_mask) = data[:7]
            obs_image, _ = get_obs_image(obs_image, kwargs["obs_type"],
                                         transform, device)
            goal_image, replaced, lang_preds = _maybe_predict_goal_with_lange3d(
                data, goal_image, device, lange3d, topopaths,
            )
            eff_goal = "image_mask_enc" if replaced else kwargs["goal_type"]
            goal_image, _ = get_goal_image(goal_image, eff_goal, transform,
                                           device, obs_image)
            dist_pred, action_pred = gnm(obs_image, goal_image)
            dist_label = dist_label.to(device)
            action_label = action_label.to(device)
            action_mask = action_mask.to(device)
            losses = _compute_losses(
                dist_label=dist_label, action_label=action_label,
                dist_pred=dist_pred, action_pred=action_pred,
                alpha=cfg["alpha"], learn_angle=cfg["learn_angle"],
                action_mask=action_mask,
            )
            l_lang_val = float("nan")
            if lang_preds is not None and lange3d_loss_fn is not None:
                gt_costs_list = [c.to(device) for c in data[7]["gt_costs_list"]]
                l_lang, _ = lange3d_loss_fn(lang_preds, gt_costs_list)
                losses["total_loss"] = losses["total_loss"] + lam * l_lang
                l_lang_val = float(l_lang.item())
            opt.zero_grad()
            losses["total_loss"].backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for g in opt.param_groups for p in g["params"]],
                    args.clip_grad,
                )
            opt.step()
            if step % max(1, args.steps // 20) == 0 or step == args.steps - 1:
                print(f"[{step:4d}/{args.steps}] "
                      f"L={losses['total_loss'].item():.4f} "
                      f"act={losses['action_loss'].item():.4f} "
                      f"lang={l_lang_val:.4f} "
                      f"cos={losses['action_waypts_cos_sim'].item():.3f}")
            step += 1
            if step >= args.steps: break
    print(f"\nTraining done in {time.time()-t0:.1f}s")

    # --- Closed-loop rollout per episode ---------------------------------
    gnm.eval();  lange3d.eval() if lange3d is not None else None
    print("\n=== Closed-loop rollout per episode ===")
    results = []
    for ei, info in enumerate(eps_info):
        traj_name = info["traj"]
        goal_time = info["goal_time"]
        # In-distribution start: earliest frame whose distance-to-goal is
        # within the training action-mask window (max_ad-1 waypoints ≈ 9).
        in_dist_start = max(
            cfg["context_size"] * spacing,
            goal_time - (max_ad - 1) * spacing,
        )
        # True "first frame" start: earliest frame with a valid context.
        ep_first_start = cfg["context_size"] * spacing

        print(f"\n  --- ep {ei}  traj={traj_name}  T={info['T']}  goal={goal_time} ---")

        all_runs = {}
        for label, start_time in [("in_window", in_dist_start),
                                  ("from_first_frame", ep_first_start)]:
            res = rollout_episode(
                base_ds, gnm, lange3d, topopaths,
                traj_name=traj_name, goal_time=goal_time,
                start_time=start_time, waypoint_spacing=spacing,
                obs_type=kwargs["obs_type"], goal_type_cfg=kwargs["goal_type"],
                transform=transform, device=device,
                metric_spacing=metric_spacing, normalize=cfg["normalize"],
                max_ad=max_ad, context_size=cfg["context_size"],
                goal_threshold=args.goal_threshold_m,
                max_steps=args.max_rollout_steps,
            )
            all_runs[label] = res
            plot_episode(res, os.path.join(args.out_dir,
                          f"ep{ei:02d}_{traj_name}_{label}.png"))
            print(f"    [{label:>16}] start={start_time:3d}  "
                  f"reached={res['reached']!s:>5}  "
                  f"final_dist={res['final_dist']:.2f}m  "
                  f"steps={res['steps_taken']}/{info['T'] - start_time}")
        results.append({"ep": ei, "traj": traj_name,
                        "T": info["T"], "goal_time": goal_time,
                        "in_window": all_runs["in_window"],
                        "from_first_frame": all_runs["from_first_frame"]})

    def _summ(key):
        n_ok = sum(1 for r in results if r[key]["reached"])
        mean_d = float(np.mean([r[key]["final_dist"] for r in results]))
        return n_ok, mean_d

    n_iw, d_iw = _summ("in_window")
    n_ff, d_ff = _summ("from_first_frame")
    print(f"\n=== Summary over {len(results)} episodes "
          f"(goal_threshold={args.goal_threshold_m}m) ===")
    print(f"  in_window         : reached {n_iw}/{len(results)}   "
          f"mean final dist = {d_iw:.3f} m")
    print(f"  from_first_frame  : reached {n_ff}/{len(results)}   "
          f"mean final dist = {d_ff:.3f} m")
    with open(os.path.join(args.out_dir, "rollout.json"), "w") as f:
        json.dump({"results": results,
                   "in_window":        {"reached": n_iw, "mean_final_dist_m": d_iw},
                   "from_first_frame": {"reached": n_ff, "mean_final_dist_m": d_ff},
                   "goal_threshold_m": args.goal_threshold_m}, f)
    print(f"Vis → {args.out_dir}/ep*.png   Log → {args.out_dir}/rollout.json")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
