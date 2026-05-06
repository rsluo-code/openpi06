import dataclasses
import glob
import pathlib

import tyro

import validation_1_hand_valuenet as single_eval
from openpi_client import websocket_client_policy as _websocket_client_policy


@dataclasses.dataclass
class Args(single_eval.Args):
    episode_dirs: list[str] = dataclasses.field(default_factory=list)
    episode_dirs_file: str | None = None
    prompt_types: list[str] = dataclasses.field(default_factory=list)


def _read_episode_dirs_file(path_str: str) -> list[str]:
    path = pathlib.Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"episode_dirs_file does not exist: {path}")
    dirs: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        dirs.append(value)
    return dirs


def _resolve_episode_dirs(args: Args) -> list[str]:
    episode_dirs: list[str] = []

    if args.episode_dirs:
        episode_dirs.extend(args.episode_dirs)

    if args.episode_dirs_file:
        episode_dirs.extend(_read_episode_dirs_file(args.episode_dirs_file))

    if args.episode_glob:
        episode_dirs.extend(
            path for path in glob.glob(args.episode_glob) if pathlib.Path(path).is_dir()
        )

    if not episode_dirs:
        episode_dirs.append(args.episode_dir)

    normalized: list[str] = []
    seen: set[str] = set()
    for episode_dir in episode_dirs:
        resolved = str(pathlib.Path(episode_dir).resolve())
        if not pathlib.Path(resolved).is_dir():
            raise FileNotFoundError(f"episode_dir does not exist: {resolved}")
        if resolved not in seen:
            seen.add(resolved)
            normalized.append(resolved)

    if not normalized:
        raise FileNotFoundError("No valid episode directories were resolved.")
    return normalized


def eval_isaac_multi(args: Args) -> None:
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    episode_dirs = _resolve_episode_dirs(args)
    prompt_types = list(args.prompt_types)
    if prompt_types and len(prompt_types) != len(episode_dirs):
        raise ValueError(
            f"prompt_types count ({len(prompt_types)}) must match episode_dirs count ({len(episode_dirs)})."
        )
    print(f"Resolved {len(episode_dirs)} episode(s) for visualization.")

    for episode_idx, episode_dir in enumerate(episode_dirs, start=1):
        print(f"\n===== [{episode_idx}/{len(episode_dirs)}] episode_dir={episode_dir} =====")
        episode_args = args
        if prompt_types:
            prompt_type = prompt_types[episode_idx - 1]
            if prompt_type not in single_eval.prompt_map:
                raise KeyError(
                    f"Unknown prompt_type={prompt_type}. Available prompt_types: {list(single_eval.prompt_map.keys())}"
                )
            episode_args = dataclasses.replace(
                args,
                prompt_type=prompt_type,
                prompt=single_eval.prompt_map[prompt_type],
            )
        single_eval._eval_single_episode(client, episode_args, episode_dir)


if __name__ == "__main__":
    tyro.cli(eval_isaac_multi)
