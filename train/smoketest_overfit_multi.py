"""
smoketest_overfit_multi.py
==========================
Overfit on a small set of N samples drawn from N different episodes,
each fixed to a deterministic (curr_time, goal_time) where goal_time is
the *last frame* of that episode (i.e. "navigate from somewhere near the
start of episode to the episode's final goal").

We pick samples whose `distance` falls inside the action-mask window so the
action loss is alive (mask=1.0) — otherwise no gradient flows.

For each episode we also run a *roll-out style* check at the end: feed the
sample through the model and report (a) cosine sim of pred vs label, (b)
end-point error of the cumulative predicted trajectory vs the GT one, and
(c) save a comparison png.
"""
from __future__ import annotations
import argparse, os, sys, json, time, types, random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from vint_train.data.vint_dataset import ViNT_Dataset
from vint_train.models.gnm.gnm import GNM
from vint_train.models.object_react.dataloader import TopoPaths
from vint_train.training.train_utils import (
    _compute_losses, get_obs_image, get_goal_image,
    _maybe_predict_goal_with_lange3d,
)
from vint_train.visualizing.action_utils import compare_waypoints_pred_to_label
from vint_train.visualizing.visualize_utils import to_numpy, numpy_to_img

from lange3dnet_train.model import LangGeoNetV2
from lange3dnet_train.losses import LangGeoNetLoss
from train import _collate_with_lange3d


class FixedSamplesDataset(Dataset):
    """Holds N pre-collected samples (tuples) and yields them in order."""
    def __init__(self, samples):
        self.samples = samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


def collect_episode_samples(base_ds, n_episodes: int,
                            len_traj_pred: int,
                            min_action_dist: int,
                            max_action_dist: int,
                            context_size: int,
                            spacing: int,
                            verbose: bool = True):
    """
    For each of the first `n_episodes` distinct trajectories, choose a
    deterministic (curr_time, goal_time) such that:
      - goal_time = last valid frame of the episode (so goal = episode end)
      - distance = (goal_time - curr_time) // spacing falls within the
        action-mask window (min_action_dist, max_action_dist) so the action
        loss is alive
    Returns a list of length `n_episodes` of fully-built sample tuples.
    """
    samples = []
    seen_trajs = set()
    for traj_name in base_ds.traj_names:
        if len(samples) >= n_episodes:
            break
        if traj_name in seen_trajs or not traj_name:
            continue
        traj = base_ds._get_trajectory(traj_name)
        T = len(traj["position"])
        # pick the largest goal_time we can; choose curr_time so distance is in
        # the middle of the action window (so the mask=1 condition holds)
        goal_time = T - 1
        target_dist = (min_action_dist + max_action_dist) // 2  # e.g. 6
        curr_time  = goal_time - target_dist * spacing
        if curr_time < context_size * spacing:
            continue
        if curr_time + len_traj_pred * spacing + 1 > T:
            continue
        distance = (goal_time - curr_time) // spacing
        if not (min_action_dist < distance < max_action_dist):
            continue

        # Patch the dataset to deterministically return our chosen goal for
        # this single sample, then load it.
        def _const_goal(self, trajectory_name, curr_t, max_d,
                        _f=traj_name, _g=goal_time):
            return _f, _g, False
        base_ds._sample_goal = types.MethodType(_const_goal, base_ds)

        # Insert the (traj, curr_time, max_dist) into the index so the dataset
        # serves it; we'll address it by appending an extra entry.
        new_idx_entry = (traj_name, curr_time, T - curr_time - 1)
        base_ds.index_to_data.append(new_idx_entry)
        i = len(base_ds.index_to_data) - 1

        sample = base_ds[i]
        if float(sample[6]) < 0.5:
            if verbose:
                print(f"  [skip] {traj_name}: action_mask=0")
            base_ds.index_to_data.pop()
            continue

        seen_trajs.add(traj_name)
        samples.append(sample)
        if verbose:
            print(f"  [{len(samples):2d}/{n_episodes}] traj={traj_name} "
                  f"T={T} curr={curr_time} goal={goal_time} "
                  f"dist={int(sample[3])} mask={float(sample[6])}")
    if len(samples) < n_episodes:
        print(f"WARNING: only collected {len(samples)} samples (asked {n_episodes})")
    return samples


def main():
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n_episodes", type=int, default=10)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lange3d_lr", type=float, default=3e-4)
    ap.add_argument("--lambda_lange3d", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--out_dir", default="./smoketest_multi")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open("config/defaults.yaml") as f: cfg = yaml.safe_load(f)
    with open(args.config) as f: cfg.update(yaml.safe_load(f))

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    print("device:", device)

    kwargs = {
        "predict_dists": cfg.get("predict_dists", True),
        "precomputed_filename": cfg.get("precomputed_filename", None),
        "pl_perturb_ratio": cfg.get("pl_perturb_ratio", 0.0),
        "pl_perturb_type": cfg.get("pl_perturb_type", "max_val"),
        "mask_crop_ratio": cfg.get("mask_crop_ratio", 1.0),
        "use_mask_grad": cfg.get("use_mask_grad", False),
        "goal_type": cfg["goal_type"], "obs_type": cfg["obs_type"],
        "dims": cfg["dims"], "goal_uses_context": cfg.get("goal_uses_context", False),
        "return_lange3d_inputs": bool(cfg.get("use_lange3d", False)),
        "clip_model_name": cfg.get("lange3d_clip_model", "openai/clip-vit-base-patch16"),
        "gnm_mask_h": cfg.get("gnm_mask_h", 60),
        "gnm_mask_w": cfg.get("gnm_mask_w", 80),
    }
    ds_cfg = cfg["datasets"]["object_react"]
    print("Building ViNT_Dataset…")
    base_ds = ViNT_Dataset(
        data_folder=ds_cfg["data_folder"],
        data_split_folder=ds_cfg["train"],
        dataset_name="object_react",
        image_size=cfg["image_size"],
        waypoint_spacing=ds_cfg["waypoint_spacing"],
        min_dist_cat=cfg["distance"]["min_dist_cat"],
        max_dist_cat=cfg["distance"]["max_dist_cat"],
        min_action_distance=cfg["action"]["min_dist_cat"],
        max_action_distance=cfg["action"]["max_dist_cat"],
        negative_mining=False,
        len_traj_pred=cfg["len_traj_pred"], learn_angle=cfg["learn_angle"],
        context_size=cfg["context_size"], context_type=cfg["context_type"],
        end_slack=ds_cfg["end_slack"], goals_per_obs=1, normalize=cfg["normalize"],
        **kwargs,
    )

    print(f"Collecting {args.n_episodes} episode samples (goal=end of ep)…")
    samples = collect_episode_samples(
        base_ds, args.n_episodes,
        len_traj_pred=cfg["len_traj_pred"],
        min_action_dist=cfg["action"]["min_dist_cat"],
        max_action_dist=cfg["action"]["max_dist_cat"],
        context_size=cfg["context_size"],
        spacing=ds_cfg["waypoint_spacing"],
    )
    n = len(samples)
    if n == 0:
        raise SystemExit("No valid episode samples collected — relax window.")
    sds = FixedSamplesDataset(samples)
    bs  = min(args.batch_size, n)
    loader = DataLoader(
        sds, batch_size=bs, shuffle=True,
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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, gnm.parameters()),
                            lr=args.lr)
    if lange3d is not None:
        lp = [p for p in lange3d.parameters() if p.requires_grad]
        if lp: opt.add_param_group({"params": lp, "lr": args.lange3d_lr})

    obs_type, goal_type_cfg = kwargs["obs_type"], kwargs["goal_type"]
    alpha, learn_angle = cfg["alpha"], cfg["learn_angle"]
    lam = float(args.lambda_lange3d)
    print(f"lambda_lange3d = {lam} | n_samples = {n} | batch_size = {bs}")

    log = []
    gnm.train()
    if lange3d is not None: lange3d.train()
    t0 = time.time()
    step = 0
    while step < args.steps:
        for data in loader:
            (obs_image, goal_image, action_label, dist_label, goal_pos,
             dataset_index, action_mask) = data[:7]
            obs_image, _ = get_obs_image(obs_image, obs_type, transform, device)
            goal_image, replaced, lang_preds = _maybe_predict_goal_with_lange3d(
                data, goal_image, device, lange3d, topopaths,
            )
            eff_goal = "image_mask_enc" if replaced else goal_type_cfg
            goal_image, _ = get_goal_image(goal_image, eff_goal, transform, device, obs_image)

            dist_pred, action_pred = gnm(obs_image, goal_image)
            dist_label, action_label, action_mask = (
                dist_label.to(device), action_label.to(device), action_mask.to(device),
            )
            losses = _compute_losses(
                dist_label=dist_label, action_label=action_label,
                dist_pred=dist_pred, action_pred=action_pred,
                alpha=alpha, learn_angle=learn_angle, action_mask=action_mask,
            )
            l_lang_val = float("nan")
            if lang_preds is not None and lange3d_loss_fn is not None:
                gt_costs_list = [c.to(device) for c in data[7]["gt_costs_list"]]
                l_lang, _ = lange3d_loss_fn(lang_preds, gt_costs_list)
                losses["total_loss"] = losses["total_loss"] + lam * l_lang
                l_lang_val = float(l_lang.item())
            opt.zero_grad()
            losses["total_loss"].backward()
            opt.step()

            entry = dict(
                step=step,
                total=float(losses["total_loss"].item()),
                action=float(losses["action_loss"].item()),
                dist=float(losses["dist_loss"].item()),
                lang=l_lang_val,
                cos=float(losses["action_waypts_cos_sim"].item()),
            )
            log.append(entry)
            if step % max(1, args.steps // 20) == 0 or step == args.steps - 1:
                print(f"[{step:4d}/{args.steps}] L={entry['total']:.4f} "
                      f"act={entry['action']:.4f} dist={entry['dist']:.3f} "
                      f"lang={entry['lang']:.4f} cos={entry['cos']:.3f}")
            step += 1
            if step >= args.steps: break
    print(f"\nDone in {time.time()-t0:.1f}s")

    # --- Per-episode evaluation -------------------------------------------
    print("\n=== Per-episode roll-out evaluation ===")
    gnm.eval()
    if lange3d is not None: lange3d.eval()
    eval_loader = DataLoader(
        sds, batch_size=1, shuffle=False,
        collate_fn=_collate_with_lange3d if kwargs["return_lange3d_inputs"] else None,
    )
    summary = []
    with torch.no_grad():
        for ei, data in enumerate(eval_loader):
            (obs_image, goal_image, action_label, dist_label, goal_pos,
             dataset_index, action_mask) = data[:7]
            obs_image, viz_obs = get_obs_image(obs_image, obs_type, transform, device)
            goal_image, replaced, _ = _maybe_predict_goal_with_lange3d(
                data, goal_image, device, lange3d, topopaths,
            )
            eff_goal = "image_mask_enc" if replaced else goal_type_cfg
            goal_image, viz_goal = get_goal_image(goal_image, eff_goal, transform, device, obs_image)
            dist_pred, action_pred = gnm(obs_image, goal_image)
            ap = action_pred[0].cpu().numpy()
            al = action_label[0].cpu().numpy()
            # endpoint error of cumulative xy
            endpoint_err = float(np.linalg.norm(ap[-1, :2] - al[-1, :2]))
            # cosine sim per step (xy only)
            denom = (np.linalg.norm(ap[:, :2], axis=1) * np.linalg.norm(al[:, :2], axis=1) + 1e-8)
            cos_per_step = (ap[:, :2] * al[:, :2]).sum(-1) / denom
            mean_cos = float(np.mean(cos_per_step))
            # average sign of x: rightward(+x positive) → if model is biased to +x always, this stays high
            sign_x = float(np.mean(np.sign(ap[:, 0])))
            summary.append(dict(
                ep=ei, dist=int(dist_label[0].item()),
                endpoint_err=endpoint_err, mean_cos=mean_cos,
                pred_xy=ap[:, :2].tolist(), label_xy=al[:, :2].tolist(),
                pred_sign_x=sign_x,
            ))
            print(f"  ep={ei:2d} dist={int(dist_label[0].item()):2d} "
                  f"endpoint_err={endpoint_err:.3f}  mean_cos_xy={mean_cos:.3f}  "
                  f"sign_x={sign_x:+.2f}")
            save_path = os.path.join(args.out_dir, f"ep{ei:02d}_after.png")
            compare_waypoints_pred_to_label(
                obs_img=numpy_to_img(to_numpy(viz_obs[0])),
                goal_img=numpy_to_img(to_numpy(viz_goal[0])),
                dataset_name="object_react",
                goal_pos=to_numpy(goal_pos[0]),
                pred_waypoints=ap, label_waypoints=al,
                save_path=save_path,
            )

    avg_endpoint = float(np.mean([s["endpoint_err"] for s in summary]))
    avg_cos      = float(np.mean([s["mean_cos"] for s in summary]))
    print(f"\nMEAN endpoint_err = {avg_endpoint:.3f}   MEAN cos = {avg_cos:.3f}")

    with open(os.path.join(args.out_dir, "log.json"), "w") as f:
        json.dump({"steps": log, "summary": summary,
                   "avg_endpoint_err": avg_endpoint, "avg_cos": avg_cos}, f)
    print(f"Vis → {args.out_dir}/ep*_after.png")
    print(f"Log → {args.out_dir}/log.json")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
