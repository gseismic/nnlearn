"""构建两个发行包，并在四个独立环境、仓库外目录验收 wheel 和 sdist。"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def run(command, cwd):
    """显式删除 Python 路径注入，确保烟测读取安装产物。"""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    print("+ " + " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, env=environment, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default="3.12")
    parser.add_argument("--constraints", type=Path, default=Path("requirements/ci-py312.txt"))
    parser.add_argument("--out-dir", type=Path, default=Path("dist/ci"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    constraints = args.constraints.resolve()
    build_constraints = root / "requirements" / "ci-build.txt"
    output = args.out_dir.resolve()
    # uv run 会把目标 Python 的目录加入 PATH；UV 保留调用者实际使用的 uv。
    uv = os.environ.get("UV") or shutil.which("uv")
    if uv is None:
        raise RuntimeError("请先安装 uv 0.9.20")

    with tempfile.TemporaryDirectory(prefix="nnlearn-distributions-") as directory:
        temporary = Path(directory)
        outside = temporary / "outside"
        outside.mkdir()
        for package, source in (("nnlearn", root), ("torch-protocol", root / "torch_protocol")):
            built = temporary / "build" / package
            run([uv, "build", source, "--python", args.python, "--build-constraints",
                 build_constraints, "--out-dir", built, "--no-build-logs"], outside)
            artifacts = sorted(built.glob("*.whl")) + sorted(built.glob("*.tar.gz"))
            assert len(artifacts) == 2, artifacts
            saved = output / package
            saved.mkdir(parents=True, exist_ok=True)
            for artifact in artifacts:
                shutil.copy2(artifact, saved / artifact.name)
                kind = "wheel" if artifact.suffix == ".whl" else "sdist"
                environment = temporary / "environments" / (package + "-" + kind)
                run([uv, "venv", "--python", args.python, environment], outside)
                interpreter = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
                run([uv, "pip", "install", "--python", interpreter, "--constraint",
                     constraints, "--build-constraints", build_constraints, artifact], outside)
                run([uv, "run", "--no-project", "--python", interpreter,
                     root / "scripts" / "smoke_installed.py", "--package", package,
                     "--source-root", root], outside)
    print("四个发行产物均已通过独立安装与仓库外烟测。", flush=True)


if __name__ == "__main__":
    main()
