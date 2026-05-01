#!/usr/bin/env python3
"""
udisksctl mount -b /dev/sda1
"""
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def log(message: str) -> None:
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")


def die(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def require_file(path: Path) -> None:
    if not path.is_file():
        die(f"Missing file: {path}")


def require_dir(path: Path) -> None:
    if not path.is_dir():
        die(f"Missing directory: {path}")


def command_exists(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def run(cmd, check=True):
    subprocess.run(cmd, check=check)


def run_capture(cmd, check=True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def normalize_downsample_label(downsample: float) -> str:
    if downsample <= 1:
        return "1"
    rounded = round(downsample)
    if abs(downsample - rounded) < 1e-8:
        return str(int(rounded))
    return str(downsample)


def count_top_level_files(path: Path) -> int:
    return sum(1 for p in path.iterdir() if p.is_file())


def get_registered_image_count(model_path: Path) -> int:
    result = run_capture(["colmap", "model_analyzer", "--path", str(model_path)], check=False)
    analyzer_out = (result.stdout or "") + "\n" + (result.stderr or "")
    match = re.search(r"Registered images:\s*(\d+)", analyzer_out)
    return int(match.group(1)) if match else -1


def select_best_sparse_model(sparse_raw: Path) -> tuple[Path, int]:
    sparse_children = sorted([p for p in sparse_raw.iterdir() if p.is_dir()])
    if not sparse_children:
        die("COLMAP mapper did not create a sparse model")

    scored_models = []
    for child in sparse_children:
        reg_count = get_registered_image_count(child)
        scored_models.append((reg_count, child))
        log(f"COLMAP sparse model candidate '{child.name}': registered_images={reg_count}")

    scored_models.sort(key=lambda x: x[0], reverse=True)
    best_reg_count, best_model = scored_models[0]
    log(f"Selected sparse model '{best_model.name}' with registered_images={best_reg_count}")
    return best_model, best_reg_count


def write_coarse_config(path: Path, scene_dir: Path, downsample: float, block_x: int, block_y: int, reorient: bool) -> None:
    content = f"""seed_everything: 42
trainer:
  accelerator: gpu
  devices: 1
  max_steps: 30000
  num_sanity_val_steps: 0
  limit_val_batches: 0
  log_every_n_steps: 1
model:
  gaussian:
    class_path: internal.models.gaussian_2d.Gaussian2D
    init_args:
      sh_degree: 2
  renderer:
    class_path: internal.renderers.sep_depth_trim_2dgs_renderer.SepDepthTrim2DGSRenderer
  metric:
    class_path: internal.metrics.citygsv2_metrics.CityGSV2Metrics
  density:
    class_path: internal.density_controllers.citygsv2_density_controller.CityGSV2DensityController
  save_ply: true
data:
  path: {scene_dir}
  parser:
    class_path: internal.dataparsers.estimated_depth_colmap_block_dataparser.EstimatedDepthBlockColmap
    init_args:
      split_mode: reconstruction
      eval_image_select_mode: ratio
      eval_ratio: 0.1
      eval_step: 8
      down_sample_factor: {downsample}
      block_dim: [{block_x}, {block_y}]
      content_threshold: 0.05
      reorient: {str(reorient).lower()}
"""
    path.write_text(content)


def write_finetune_config(path: Path, scene_dir: Path, downsample: float, block_x: int, block_y: int, reorient: bool, coarse_ckpt: str) -> None:
    content = f"""seed_everything: 42
trainer:
  accelerator: gpu
  devices: 1
  max_steps: 30000
  num_sanity_val_steps: 0
  limit_val_batches: 0
  log_every_n_steps: 1
model:
  gaussian:
    class_path: internal.models.gaussian_2d.Gaussian2D
    init_args:
      sh_degree: 2
      optimization:
        means_lr_init: 0.000064
        means_lr_scheduler:
          lr_final: 0.00000064
        scales_lr: 0.004
  renderer:
    class_path: internal.renderers.sep_depth_trim_2dgs_renderer.SepDepthTrim2DGSRenderer
  metric:
    class_path: internal.metrics.citygsv2_metrics.CityGSV2Metrics
  density:
    class_path: internal.density_controllers.citygsv2_density_controller.CityGSV2DensityController
  initialize_from: {coarse_ckpt}
  save_ply: true
data:
  path: {scene_dir}
  parser:
    class_path: internal.dataparsers.estimated_depth_colmap_block_dataparser.EstimatedDepthBlockColmap
    init_args:
      split_mode: reconstruction
      eval_image_select_mode: ratio
      eval_ratio: 0.1
      eval_step: 8
      down_sample_factor: {downsample}
      block_dim: [{block_x}, {block_y}]
      content_threshold: 0.05
      reorient: {str(reorient).lower()}
"""
    path.write_text(content)


def parse_args():
    parser = argparse.ArgumentParser(description="CityGaussian custom-scene pipeline with practical safeguards for small scenes.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--downsample", type=float, default=2)
    parser.add_argument("--block-x", type=int, default=1)
    parser.add_argument("--block-y", type=int, default=1)
    parser.add_argument("--matcher", choices=["exhaustive", "sequential"], default="sequential")
    parser.add_argument("--single-camera", type=int, choices=[0, 1], default=1)
    parser.add_argument("--reorient", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    repo_root = Path.cwd()
    images_src = Path(args.images)
    #require_dir(images_src)

    require_file(repo_root / "main.py")
    require_dir(repo_root / "configs")
    require_dir(repo_root / "utils")

    if not command_exists("python"):
        die("python not found")
    if not command_exists("colmap"):
        die("colmap not found")

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("WANDB_MODE", "offline")

    scene = args.scene
    scene_dir = repo_root / "data" / scene
    colmap_work_dir = scene_dir / "_colmap_work"
    coarse_name = f"{scene}_coarse"
    finetune_name = f"{scene}_finetune"
    coarse_cfg = repo_root / "configs" / f"{coarse_name}.yaml"
    finetune_cfg = repo_root / "configs" / f"{finetune_name}.yaml"

    scene_dir.mkdir(parents=True, exist_ok=True)
    scene_images_dir = scene_dir / "images"
    if scene_images_dir.exists():
        shutil.rmtree(scene_images_dir)
    if str(images_src).lower().endswith('.mp4'):
        os.makedirs(scene_images_dir, exist_ok=True)
        # sample frames to images
        import cv2 as cv 
        cap = cv.VideoCapture(images_src)
        sample_every_x_frames = 12
        for frame_idx in range(int(cap.get(cv.CAP_PROP_FRAME_COUNT))):
            _, frame = cap.read()
            if frame is not None and frame_idx % sample_every_x_frames == 0:
                cv.imwrite(os.path.join(scene_images_dir, f"frame_{str(frame_idx).zfill(6)}.jpg"),frame)
    else:
        shutil.copytree(images_src, scene_images_dir)

    image_count = count_top_level_files(scene_images_dir)
    if image_count <= 0:
        die(f"No images found in {scene_images_dir}")

    downsample_label = normalize_downsample_label(args.downsample)

    log("Step 1/9: COLMAP feature extraction")
    colmap_work_dir.mkdir(parents=True, exist_ok=True)
    db_path = colmap_work_dir / "database.db"
    if db_path.exists():
        db_path.unlink()
    run([
        "colmap", "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(scene_images_dir),
        "--ImageReader.single_camera", str(args.single_camera),
    ])

    log("Step 2/9: COLMAP matching")
    if args.matcher == "sequential":
        run(["colmap", "sequential_matcher", "--database_path", str(db_path)])
    else:
        run(["colmap", "exhaustive_matcher", "--database_path", str(db_path)])

    log("Step 3/9: COLMAP sparse reconstruction")
    sparse_raw = colmap_work_dir / "sparse_raw"
    if sparse_raw.exists():
        shutil.rmtree(sparse_raw)
    sparse_raw.mkdir(parents=True, exist_ok=True)
    run([
        "colmap", "mapper",
        "--database_path", str(db_path),
        "--image_path", str(scene_images_dir),
        "--output_path", str(sparse_raw),
    ])

    sparse_raw_dir, registered_images = select_best_sparse_model(sparse_raw)
    if registered_images <= 0:
        die("Could not parse COLMAP registered image count from mapper output models")

    registration_ratio = registered_images / image_count
    log(f"Sparse-model registration quality: {registered_images}/{image_count} images ({registration_ratio:.1%})")
    if registration_ratio < 0.6:
        log(
            "WARNING: Low registration ratio. Results will likely be poor. "
            "Try --matcher sequential (video order), keep strong overlap, remove blurry frames, "
            "and verify camera intrinsics with --single-camera 1."
        )

    log("Step 4/9: Image undistortion")
    dense_dir = colmap_work_dir / "dense"
    if dense_dir.exists():
        shutil.rmtree(dense_dir)
    run([
        "colmap", "image_undistorter",
        "--image_path", str(scene_images_dir),
        "--input_path", str(sparse_raw_dir),
        "--output_path", str(dense_dir),
    ])

    sparse_target = scene_dir / "sparse"
    if sparse_target.exists():
        shutil.rmtree(sparse_target)
    (sparse_target / "0").mkdir(parents=True, exist_ok=True)
    for item in (dense_dir / "sparse").iterdir():
        dst = sparse_target / "0" / item.name
        if item.is_dir():
            shutil.copytree(item, dst)
        else:
            shutil.copy2(item, dst)

    log("Sparse-model summary")
    run(["colmap", "model_analyzer", "--path", str(sparse_target / "0")], check=False)

    log("Step 5/9: Downsampling undistorted images")
    if downsample_label == "1":
        if scene_images_dir.exists():
            shutil.rmtree(scene_images_dir)
        shutil.copytree(dense_dir / "images", scene_images_dir)
    else:
        run([
            "python", "utils/image_downsample.py",
            str(dense_dir / "images"),
            "--dst", str(scene_dir / f"images_{downsample_label}"),
            "--factor", str(args.downsample),
        ])
        alias = scene_dir / f"images_{args.downsample}"
        if not alias.exists():
            try:
                alias.symlink_to(f"images_{downsample_label}")
            except Exception:
                pass

    log("Step 6/9: Estimating monocular depths on the same image directory used for training")
    est_depths = scene_dir / "estimated_depths"
    est_scales = scene_dir / "estimated_depth_scales.json"
    if est_depths.exists():
        shutil.rmtree(est_depths)
    if est_scales.exists():
        est_scales.unlink()

    run([
        "python", "utils/estimate_dataset_depths.py",
        str(scene_dir),
        "--image_dir", f"images_{downsample_label}",
    ])

    log("Step 7/9: Writing CityGaussian coarse config")
    write_coarse_config(coarse_cfg, scene_dir, args.downsample, args.block_x, args.block_y, args.reorient)

    log("Step 8/9: Training coarse model")
    run(["python", "main.py", "fit", "--config", str(coarse_cfg), "-n", coarse_name])

    coarse_ckpts = sorted(glob.glob(str(repo_root / "outputs" / coarse_name / "checkpoints" / "*.ckpt")))
    if not coarse_ckpts:
        die("Could not find coarse checkpoint")
    coarse_ckpt = coarse_ckpts[-1]
    log(f"Coarse checkpoint: {coarse_ckpt}")

    log("Step 8.5/9: Writing CityGaussian finetune config")
    write_finetune_config(finetune_cfg, scene_dir, args.downsample, args.block_x, args.block_y, args.reorient, coarse_ckpt)

    final_model_path = repo_root / "outputs" / coarse_name
    used_finetune = False

    log("Step 9/9: Partition, finetune if safe, merge, and mesh extraction")
    if image_count <= 50:
        log(f"Scene has {image_count} images (<= 50); skipping partition finetune and using the coarse model as final output.")
    else:
        run(["python", "utils/partition_citygs.py", "--config_path", str(finetune_cfg), "--force"])

        part_dir = scene_dir / "partition" / f"partitions-dim_{args.block_x}_{args.block_y}_visibility_0.05"
        require_dir(part_dir)

        part_txts = sorted(part_dir.glob("*.txt"))
        if not part_txts:
            die(f"No partition files found in {part_dir}")

        too_small = False
        for part_txt in part_txts:
            with part_txt.open("r") as f:
                count = sum(1 for line in f if line.strip())
            if count <= 50:
                log(f"Partition {part_txt.name} has {count} images (<= 50); skipping finetune and using the coarse model as final output.")
                too_small = True
                break

        if not too_small:
            num_blocks = args.block_x * args.block_y
            log(f"Single-GPU mode: running {num_blocks} partition(s) sequentially on CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
            for block_id in range(num_blocks):
                run([
                    "python", "main.py", "fit",
                    "--config", str(finetune_cfg),
                    "--data.parser.block_id", str(block_id),
                    "--name", finetune_name,
                    "--project", scene,
                    "--logger", "wandb",
                ])

            run(["python", "utils/merge_citygs_ckpts.py", str(repo_root / "outputs" / finetune_name)])
            final_model_path = repo_root / "outputs" / finetune_name
            used_finetune = True

    log(f"Mesh extraction from {final_model_path}")
    mesh_cmd = ["python", "utils/gs2d_mesh_extraction.py", str(final_model_path), "--dataset_path", str(scene_dir)]
    mesh_result = subprocess.run(mesh_cmd)
    if mesh_result.returncode != 0:
        log("Default mesh post-processing failed; retrying with --num_cluster 1")
        run(mesh_cmd + ["--num_cluster", "1"])

    log("Summary")
    print(f"Scene: {scene}")
    print(f"Input images: {image_count}")
    print(f"Used finetune: {str(used_finetune).lower()}")
    print(f"Final model path: {final_model_path}")
    if (final_model_path / "fuse.ply").is_file():
        print(f"Mesh: {final_model_path / 'fuse.ply'}")
    if (final_model_path / "fuse_post.ply").is_file():
        print(f"Postprocessed mesh: {final_model_path / 'fuse_post.ply'}")

    print("\nInteractive model viewer:")
    print(f"  python viewer.py {final_model_path} --port 8080")
    print("\nFor your last failed coarse extraction, this also works now:")
    print(f"  python utils/gs2d_mesh_extraction.py outputs/{coarse_name} --dataset_path {scene_dir} --num_cluster 1")


if __name__ == "__main__":
    main()
