import os
import argparse
import yaml
import add_pypath
import subprocess
import traceback
import time
import selectors
import functools

from tqdm.auto import tqdm
from argparser_utils import split_stoppable_args, parser_stoppable_args
from internal.utils.general_utils import parse


def get_project_output_dir_by_name(project_name: str) -> str:
    return os.path.join(os.path.join(os.path.dirname(os.path.dirname(__file__))), "outputs", project_name)


def srun_output_dir(project_name: str) -> str:
    return os.path.join(get_project_output_dir_by_name(project_name), "srun-outputs")


def run_subprocess(args, output_redirect) -> int:
    sel = selectors.DefaultSelector()
    with subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as p:
        sel.register(p.stdout, selectors.EVENT_READ)
        sel.register(p.stderr, selectors.EVENT_READ)

        while True:
            if len(sel.get_map()) == 0:
                break

            events = sel.select()
            for key, mask in events:
                line = key.fileobj.readline()
                if len(line) == 0:
                    sel.unregister(key.fileobj)
                    continue
                output_redirect(line.decode("utf-8").rstrip("\n"))

        p.wait()
        return p.returncode


def train_a_partition(
    config_args,
    extra_training_args,
    srun_args,
    partition_idx,
    gpu_id,
):
    config_file = os.path.join(config_args.config_dir, f"{config_args.config_name}.yaml")
    project_name = config_args.project_name
    dry_run = config_args.dry_run

    args = [
        "python",
        "main.py",
        "fit",
        "--config",
        config_file,
        "--data.parser.block_id",
        str(partition_idx),
    ]
    args += extra_training_args

    experiment_name = config_args.config_name
    args += [
        f"-n={experiment_name}",
        "--project",
        project_name,
        "--logger",
        "wandb",
    ]

    print_func = print
    run_func = functools.partial(
        subprocess.run,
        env=dict(**os.environ, CUDA_VISIBLE_DEVICES=str(gpu_id)),
    )

    if len(srun_args) > 0:
        def tqdm_write(i):
            tqdm.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] #{partition_idx}: {i}")

        def run_with_tqdm_write(cmd_args):
            return run_subprocess(cmd_args, tqdm_write)

        run_func = run_with_tqdm_write
        print_func = tqdm_write
        output_filename = os.path.join(srun_output_dir(config_args.config_name), f"block_{partition_idx}.txt")
        args = [
            "srun",
            f"--output={output_filename}",
            f"--job-name={config_args.project_name}-{experiment_name}",
        ] + srun_args + args

    ret_code = -1
    if dry_run:
        print(" \\\n ".join(args))
    else:
        try:
            print_func(str(args))
            result = run_func(args)
            ret_code = result if isinstance(result, int) else result.returncode
        except KeyboardInterrupt as e:
            raise e
        except Exception:
            traceback.print_exc()

    return partition_idx, ret_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", "-n", type=str, required=True)
    parser.add_argument("--config_dir", "-c", type=str, default="./configs")
    parser.add_argument("--project_name", "-p", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true", default=False)
    args, training_and_srun_args = parser_stoppable_args(parser)
    training_args, srun_args = split_stoppable_args(training_and_srun_args)

    if args.project_name is None:
        args.project_name = args.config_name

    config_path = os.path.join(args.config_dir, f"{args.config_name}.yaml")
    with open(config_path, 'r') as f:
        config = parse(yaml.load(f, Loader=yaml.FullLoader))

    num_blocks = config.data.parser.init_args.block_dim[0] * config.data.parser.init_args.block_dim[1]

    # Always run partitions sequentially on a single visible GPU.
    # Respect CUDA_VISIBLE_DEVICES from the shell; inside each subprocess use device index 0.
    if len(srun_args) == 0:
        print(f"Single-GPU mode: running {num_blocks} partition(s) sequentially on CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '0')}")
        for block_id in range(num_blocks):
            finished_idx, ret_code = train_a_partition(args, training_args, srun_args, block_id, 0)
            if ret_code != 0:
                raise SystemExit(ret_code)
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] #{finished_idx} exited with code {ret_code} | {block_id + 1}/{num_blocks}")
    else:
        print("SLURM mode enabled; launching one partition at a time")
        os.makedirs(srun_output_dir(args.config_name), exist_ok=True)
        for block_id in range(num_blocks):
            finished_idx, ret_code = train_a_partition(args, training_args, srun_args, block_id, 0)
            if ret_code != 0:
                raise SystemExit(ret_code)
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] #{finished_idx} exited with code {ret_code} | {block_id + 1}/{num_blocks}")
