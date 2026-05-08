import argparse
import copy
import itertools
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

import yaml


VALID_ACTIVATIONS = {"softplus", "relu", "gelu", "silu", "leaky_relu"}
VALID_TRANSFORMS = {"none", "log1p", "sqrt", "cbrt", "minmax", "log1p_minmax", "clamp99_minmax", "tanh100", "tanh500"}


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def write_yaml(path, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def deep_merge(base, override):
    result = copy.deepcopy(base)

    for key, value in (override or {}).items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)

    return result


def set_nested(cfg, dotted_key, value):
    cursor = cfg
    parts = dotted_key.split(".")

    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]

    cursor[parts[-1]] = copy.deepcopy(value)


def iter_grid(grid):
    keys = list(grid)
    values = [grid[key] for key in keys]

    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def model_filename(cfg):
    return (
        f"model_{cfg['model']['type']}"
        f"_latent{cfg['model']['latent']}"
        f"_ch{'_'.join(str(c) for c in cfg['model']['channels'])}"
        f"_beta{cfg['train']['beta']}"
        f"_lr{cfg['optimizer']['lr']}"
        f"_epoch{cfg['train']['epochs']}"
        f"_act{cfg['model']['activation']}"
        f"_kern{cfg['model']['kernel']}"
        f"_stride{cfg['model']['stride']}"
        f"_pad{cfg['model']['padding']}"
        f"_hw{'x'.join(str(d) for d in cfg['model']['input_hw'])}"
        f"_tx{cfg['data'].get('transform', 'none')}.pt"
    )


def validate_training_config(cfg):
    activation = cfg["model"]["activation"]
    if activation not in VALID_ACTIVATIONS:
        valid = ", ".join(sorted(VALID_ACTIVATIONS))
        raise ValueError(
            f"unknown model.activation {activation!r}; expected one of: {valid}"
        )

    transform = cfg.get("data", {}).get("transform", "none")
    if transform not in VALID_TRANSFORMS:
        valid = ", ".join(sorted(VALID_TRANSFORMS))
        raise ValueError(
            f"unknown data.transform {transform!r}; expected one of: {valid}"
        )

    input_h, input_w = cfg["model"]["input_hw"]
    channels = cfg["model"]["channels"]
    scale = 2 ** len(channels)

    if input_h % scale != 0 or input_w % scale != 0:
        raise ValueError(
            "model.input_hw must be divisible by 2 ** len(model.channels) "
            f"for the current VAE decoder; got input_hw={cfg['model']['input_hw']} "
            f"and {len(channels)} channel blocks"
        )

    if input_h // scale < 1 or input_w // scale < 1:
        raise ValueError(
            f"encoded feature map would be empty: input_hw={cfg['model']['input_hw']}, "
            f"channels={channels}"
        )


def resolve_local_path(path, fallback_base=None):
    candidate = Path(path).expanduser()
    if candidate.is_absolute() or candidate.exists():
        return candidate

    if fallback_base is not None:
        fallback = fallback_base / candidate
        if fallback.exists():
            return fallback

    return candidate


def shell_join(parts):
    return " ".join(shlex.quote(str(part)) for part in parts)


def remote_path_arg(host, path):
    return f"{host}:{shlex.quote(str(path))}"


def run(cmd, dry_run=False):
    print("$", shell_join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def ssh(host, command, dry_run=False):
    run(["ssh", host, command], dry_run=dry_run)


def rsync_to_remote(src, host, dst, dry_run=False):
    run(["rsync", "-avz", str(src), remote_path_arg(host, dst)], dry_run=dry_run)


def rsync_from_remote(host, src, dst_dir, dry_run=False):
    run(["rsync", "-avz", remote_path_arg(host, src), str(dst_dir) + "/"], dry_run=dry_run)


def remote_file_exists(host, path, dry_run=False):
    command = f"test -f {shlex.quote(str(path))}"
    print("$", shell_join(["ssh", host, command]))
    if dry_run:
        return False

    result = subprocess.run(["ssh", host, command], check=False)
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False

    result.check_returncode()
    return False


def remote_command_output(host, command):
    return subprocess.check_output(["ssh", host, command], text=True)


def free_remote_gpus(host, min_free_gb):
    query = (
        "nvidia-smi "
        "--query-gpu=index,memory.free "
        "--format=csv,noheader,nounits"
    )
    output = remote_command_output(host, query)
    free_gpus = []
    min_free_mb = min_free_gb * 1024

    for line in output.splitlines():
        if not line.strip():
            continue
        index, free_mb = [part.strip() for part in line.split(",", maxsplit=1)]
        if int(free_mb) >= min_free_mb:
            free_gpus.append(index)

    return free_gpus


def parse_gpu_devices(value, host, min_free_gb, dry_run):
    if not value:
        return []

    if value == "auto":
        print(
            "$",
            shell_join(
                [
                    "ssh",
                    host,
                    "nvidia-smi --query-gpu=index,memory.free "
                    "--format=csv,noheader,nounits",
                ]
            ),
        )
        if dry_run:
            return ["0", "1"]
        return free_remote_gpus(host, min_free_gb)

    return [device.strip() for device in value.split(",") if device.strip()]


def require(mapping, key, source):
    value = mapping.get(key)
    if value in (None, ""):
        raise ValueError(f"{source} must define {key!r}")
    return value


def resolve_remote_data_path(cfg, data_src):
    data_path = str(cfg["data"]["path"])
    if data_path.endswith("/"):
        cfg["data"]["path"] = str(PurePosixPath(data_path) / data_src.name)


def prepare_runs(sweep, base_cfg, remote_overrides, data_src, start_index, limit):
    base_remote_cfg = deep_merge(base_cfg, remote_overrides)
    resolve_remote_data_path(base_remote_cfg, data_src)

    all_combos = list(iter_grid(sweep.get("grid", {})))
    selected = [
        (index, combo)
        for index, combo in enumerate(all_combos, start=1)
        if index >= start_index
    ]

    if limit is not None:
        selected = selected[:limit]

    runs = []
    skipped = []

    for index, combo in selected:
        cfg = copy.deepcopy(base_remote_cfg)
        for key, value in combo.items():
            set_nested(cfg, key, value)
        cfg["sweep"] = {"index": index, "parameters": combo}

        try:
            validate_training_config(cfg)
        except ValueError as exc:
            skipped.append((index, combo, str(exc)))
            continue

        runs.append((index, combo, cfg, model_filename(cfg)))

    return runs, skipped, len(all_combos), len(selected)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a remote VAE training sweep one configuration at a time."
    )
    parser.add_argument("--sweep", default="configs/sweep.yaml")
    parser.add_argument("--remote", default="configs/remote.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--force-data", action="store_true")
    parser.add_argument("--skip-split", action="store_true")
    parser.add_argument(
        "--gpu-devices",
        help="Comma-separated remote GPU ids, or 'auto' to use GPUs with enough free memory.",
    )
    parser.add_argument(
        "--min-free-gpu-gb",
        type=int,
        default=20,
        help="Minimum free VRAM per GPU for --gpu-devices auto.",
    )
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip runs whose model already exists locally.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue after a failed remote run.",
    )
    parser.add_argument(
        "--keep-remote-model",
        action="store_true",
        help="Do not delete remote .pt files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    sweep_path = resolve_local_path(args.sweep)
    sweep = load_yaml(sweep_path)

    base_path = resolve_local_path(sweep["base"], fallback_base=sweep_path.parent)
    base_cfg = load_yaml(base_path)

    remote_path = resolve_local_path(args.remote)
    remote_profile = load_yaml(remote_path)
    remote_settings = remote_profile.get("remote", {})
    remote_overrides = {
        key: value for key, value in remote_profile.items() if key != "remote"
    }

    host = require(remote_settings, "host", args.remote)
    project_dir = PurePosixPath(require(remote_settings, "project_dir", args.remote))
    local_model_dir = Path(
        require(remote_settings, "local_model_dir", args.remote)
    ).expanduser()
    data_src = Path(require(remote_settings, "data_src", args.remote)).expanduser()
    splits_src_dir = Path(
        remote_settings.get("splits_src_dir", base_cfg["output"]["splits_dir"])
    ).expanduser()
    python_cmd = remote_settings.get("python", "uv run python")
    remote_config_dir = PurePosixPath(
        remote_settings.get("config_dir", str(project_dir / "configs" / "generated_sweeps"))
    )
    delete_remote_model = (
        remote_settings.get("delete_remote_model", True)
        and not args.keep_remote_model
    )

    if not args.dry_run and not args.skip_data and not data_src.exists():
        raise FileNotFoundError(f"remote.data_src does not exist: {data_src}")

    runs, skipped, total_grid, selected_count = prepare_runs(
        sweep=sweep,
        base_cfg=base_cfg,
        remote_overrides=remote_overrides,
        data_src=data_src,
        start_index=args.start_index,
        limit=args.limit,
    )

    print(
        f"Prepared {len(runs)} valid run(s) from {selected_count} selected "
        f"combination(s); full grid has {total_grid} combination(s)."
    )

    for index, combo, reason in skipped:
        print(f"Skipping run {index}: {combo} ({reason})")

    if not runs:
        return 0

    if not args.dry_run:
        local_model_dir.mkdir(parents=True, exist_ok=True)
        (local_model_dir / "sweep_configs").mkdir(parents=True, exist_ok=True)

    mkdir_paths = {
        remote_config_dir,
        project_dir / "logs",
    }

    first_cfg = runs[0][2]
    data_dst = PurePosixPath(first_cfg["data"]["path"])
    mkdir_paths.add(data_dst.parent)

    for _, _, cfg, _ in runs:
        mkdir_paths.add(PurePosixPath(cfg["output"]["dir"]))
        mkdir_paths.add(PurePosixPath(cfg["output"]["splits_dir"]))

    ssh(host, f"test -d {shlex.quote(str(project_dir))}", dry_run=args.dry_run)
    ssh(host, "mkdir -p " + shell_join(sorted(mkdir_paths)), dry_run=args.dry_run)

    if not args.skip_data:
        if args.force_data or not remote_file_exists(host, data_dst, dry_run=args.dry_run):
            rsync_to_remote(data_src, host, data_dst, dry_run=args.dry_run)
        else:
            print(f"Remote data exists, skipping upload: {host}:{data_dst}")

    proton_keys = {cfg["data"]["proton"] for _, _, cfg, _ in runs}
    if not args.skip_split:
        for proton_key in sorted(proton_keys):
            split_name = f"split_{proton_key}.npz"
            split_src = splits_src_dir / split_name
            split_dst = PurePosixPath(runs[0][2]["output"]["splits_dir"]) / split_name

            if not args.dry_run and not split_src.exists():
                raise FileNotFoundError(
                    f"local split file does not exist: {split_src}. "
                    "Use --skip-split to let the remote create a new split."
                )

            rsync_to_remote(split_src, host, split_dst, dry_run=args.dry_run)

    gpu_devices = parse_gpu_devices(
        args.gpu_devices,
        host=host,
        min_free_gb=args.min_free_gpu_gb,
        dry_run=args.dry_run,
    )

    if args.gpu_devices and not gpu_devices:
        raise RuntimeError(
            f"No GPUs have at least {args.min_free_gpu_gb} GB free on {host}."
        )

    def run_one(ordinal, run_info, gpu_device=None):
        index, combo, cfg, name = run_info
        prefix = f"[{ordinal}/{len(runs)}]"
        gpu_label = f" gpu={gpu_device}" if gpu_device is not None else ""
        print(f"\n{prefix}{gpu_label} run {index}: {combo}")

        local_model_path = local_model_dir / name
        if args.resume and local_model_path.exists():
            print(f"Local model exists, skipping: {local_model_path}")
            return None

        local_cfg_path = (
            local_model_dir
            / "sweep_configs"
            / f"run_{index:04d}_{Path(name).stem}.yaml"
        )
        remote_cfg_path = remote_config_dir / local_cfg_path.name
        remote_model_path = PurePosixPath(cfg["output"]["dir"]) / name
        remote_curve_path = remote_model_path.with_name(name.replace(".pt", "_curves.png"))
        remote_log_path = project_dir / "logs" / name.replace(".pt", ".json")

        try:
            if not args.dry_run:
                write_yaml(local_cfg_path, cfg)

            rsync_to_remote(local_cfg_path, host, remote_cfg_path, dry_run=args.dry_run)

            train_command = (
                f"cd {shlex.quote(str(project_dir))} && "
                f"PYTHONPATH={shlex.quote(str(project_dir))} "
                f"{python_cmd} scripts/run_training.py "
                f"--config {shlex.quote(str(remote_cfg_path))}"
            )
            if gpu_device is not None:
                train_command = (
                    f"cd {shlex.quote(str(project_dir))} && "
                    f"CUDA_VISIBLE_DEVICES={shlex.quote(str(gpu_device))} "
                    f"PYTHONPATH={shlex.quote(str(project_dir))} "
                    f"{python_cmd} scripts/run_training.py "
                    f"--config {shlex.quote(str(remote_cfg_path))}"
                )
            ssh(host, train_command, dry_run=args.dry_run)

            rsync_from_remote(host, remote_model_path, local_model_dir, dry_run=args.dry_run)
            rsync_from_remote(host, remote_curve_path, local_model_dir, dry_run=args.dry_run)
            rsync_from_remote(host, remote_log_path, local_model_dir, dry_run=args.dry_run)

            if delete_remote_model:
                ssh(host, f"rm -f {shlex.quote(str(remote_model_path))}", dry_run=args.dry_run)

            return None

        except subprocess.CalledProcessError as exc:
            print(f"Run {index} failed with exit code {exc.returncode}.")
            return index, combo, exc.returncode

    failures = []

    if gpu_devices:
        print(f"Using remote GPU devices concurrently: {', '.join(gpu_devices)}")
        with ThreadPoolExecutor(max_workers=len(gpu_devices)) as executor:
            futures = []
            for ordinal, run_info in enumerate(runs, start=1):
                gpu_device = gpu_devices[(ordinal - 1) % len(gpu_devices)]
                futures.append(executor.submit(run_one, ordinal, run_info, gpu_device))

            for future in as_completed(futures):
                failure = future.result()
                if failure is not None:
                    failures.append(failure)
                    if not args.keep_going:
                        break
    else:
        for ordinal, run_info in enumerate(runs, start=1):
            failure = run_one(ordinal, run_info)
            if failure is not None:
                failures.append(failure)
                if not args.keep_going:
                    break

    if failures:
        print("\nFailures:")
        for index, combo, returncode in failures:
            print(f"  run {index}: returncode={returncode}, parameters={combo}")
        return 1

    print("\nSweep finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
