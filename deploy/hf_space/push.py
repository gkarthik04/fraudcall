"""Stage the small set of files the web app needs and (optionally) upload them to a Hugging Face Space.

  python -m deploy.hf_space.push --stage-only            # build deploy/_stage and stop (no account needed)
  python -m deploy.hf_space.push --repo YOUR_NAME/fraudcall   # stage, create the Space if needed, upload

Upload needs a Hugging Face login:  hf auth login   (you enter the token yourself; this script never reads or prints it).
The Space builds the Docker image on Hugging Face's servers, so nothing heavy is installed on this computer.
"""
import argparse, json, os, shutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
STAGE = os.path.join(ROOT, "deploy", "_stage")


def cp(src, dst=None):
    dst = dst or src
    s = os.path.join(ROOT, src); d = os.path.join(STAGE, dst)
    os.makedirs(os.path.dirname(d), exist_ok=True)
    if os.path.isdir(s):
        shutil.copytree(s, d, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    else:
        shutil.copy2(s, d)


def stage():
    if os.path.isdir(STAGE):
        shutil.rmtree(STAGE)
    os.makedirs(STAGE)
    here = os.path.join("deploy", "hf_space")
    cp(os.path.join(here, "Dockerfile"), "Dockerfile"); cp(os.path.join(here, "requirements.txt"), "requirements.txt"); cp(os.path.join(here, "SPACE_README.md"), "README.md")
    for d in ("src", "web", "figures"):
        cp(d)
    cp("configs/base.yaml")
    for f in ("results/demo_config.json", "results/demo_samples.json", "results/summary_by_cond.csv", "results/wer_by_cond.csv", "data/features/vocab.json"):
        cp(f)
    dc = json.load(open(os.path.join(ROOT, "results/demo_config.json"), encoding="utf8"))
    for spec in dc["models"].values():
        cp(spec["ckpt"])
    for s in json.load(open(os.path.join(ROOT, "results/demo_samples.json"), encoding="utf8")):
        cp(s["wav"])
    size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(STAGE) for f in fs)
    print(f"staged {STAGE}  ({size / 1e6:.1f} MB)")
    return STAGE


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--repo"); ap.add_argument("--stage-only", action="store_true"); ap.add_argument("--private", action="store_true")
    a = ap.parse_args()
    stage()
    if a.stage_only:
        return
    if not a.repo:
        raise SystemExit("give --repo YOUR_NAME/fraudcall (or use --stage-only)")
    from huggingface_hub import HfApi
    api = HfApi()
    who = api.whoami()                              # raises if not logged in
    print("logged in as", who["name"])
    api.create_repo(a.repo, repo_type="space", space_sdk="docker", private=a.private, exist_ok=True)
    api.upload_folder(folder_path=STAGE, repo_id=a.repo, repo_type="space", commit_message="Deploy FraudCall web app")
    print("uploaded. Build and start take a few minutes: https://huggingface.co/spaces/" + a.repo)


if __name__ == "__main__":
    main()
