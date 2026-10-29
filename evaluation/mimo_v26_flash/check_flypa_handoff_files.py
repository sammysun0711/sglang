"""Check the consolidated handoff using the standard library only; no GPU calls."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--dependencies", action="store_true")
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--aiter-root", type=Path)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--model-root", type=Path, default=Path(os.environ.get("MODEL_PATH", "/models/MiMo-V2.6-Flash-MOPD")))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    workspace = args.workspace or Path(os.environ.get("MIMO_ROOT", str(repo.parent)))
    manifest = json.loads((repo / "evaluation/mimo_v26_flash/flypa_handoff_assets/file-manifest.json").read_text())
    missing, mismatched, checked = [], [], []

    def check(path, sha256=None):
        if not path.is_file():
            missing.append(str(path))
            return
        checked.append(str(path))
        if sha256 and digest(path) != sha256:
            mismatched.append(str(path))

    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True)
    ancestry = subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", manifest["sglang_base"], "HEAD"], capture_output=True)
    if head.returncode or ancestry.returncode:
        mismatched.append(f"SGLang base {manifest['sglang_base']} is not an ancestor of {head.stdout.strip() or 'the missing checkout'}")
    for item in manifest["files"]:
        check(repo / item["path"], item["sha256"])

    external = {}
    if args.dependencies:
        aiter = args.aiter_root or Path(os.environ.get("AITER_ROOT", str(workspace / "aiter")))
        ck = aiter / "3rdparty/composable_kernel"
        runtime = args.runtime_root or workspace / "runtime724"
        dataset = args.dataset or Path(os.environ.get("MIMO_DATASET", str(workspace / "sglang/evaluation/data/ShareGPT_V3_unfiltered_cleaned_split.json")))
        for name, root, expected in (("aiter", aiter, manifest["dependencies"]["aiter_head"]),
                                     ("composable_kernel", ck, manifest["dependencies"]["ck_head"])):
            result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
            external[name] = dict(path=str(root), head=result.stdout.strip())
            ancestor = subprocess.run(["git", "-C", str(root), "merge-base", "--is-ancestor", expected, "HEAD"], capture_output=True)
            if result.returncode or ancestor.returncode:
                mismatched.append(f"{name} base {expected} is not an ancestor of {result.stdout.strip() or 'the missing checkout'}")
        for item in manifest["dependencies"]["aiter_files"]:
            check(aiter / item["path"], item["sha256"])
        for item in manifest["dependencies"]["ck_files"]:
            check(ck / item["path"], item["sha256"])
        for item in manifest["dependencies"]["runtime_files"]:
            check(runtime / item["path"], item["sha256"])
        check(runtime / "python/torch/__init__.py")
        check(dataset, manifest["dependencies"]["dataset"]["sha256"])
        external["runtime"] = str(runtime)
        external["dataset"] = str(dataset)
        external["model"] = str(args.model_root)
        for directory in (args.model_root, args.model_root / "dflash"):
            check(directory / "config.json")
            index = directory / "model.safetensors.index.json"
            if index.is_file():
                check(index)
                for shard in set(json.loads(index.read_text())["weight_map"].values()):
                    check(directory / shard)
            else:
                check(directory / "model.safetensors")
        check(args.model_root / "tokenizer.json")

    result = dict(passed=not missing and not mismatched, repo=str(repo),
                  checked_files=len(checked), missing=missing, mismatched=mismatched,
                  dependencies_checked=args.dependencies, external=external,
                  scope="File presence, code/data hashes and repository bases only. No GPU import, numerical test or E2E run; model shards checked for presence only.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
