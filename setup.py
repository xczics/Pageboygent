from pathlib import Path
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop


def _export_config_docs() -> None:
    """Generate config_table.md, .env.example, and pageboygent.toml.example after install."""
    try:
        from Pageboygent.config import export_config_docs, export_config_examples #type: ignore
        root = Path(__file__).parent
        export_config_docs(output_path=root / "config_table.md")
        print(f"[pageboygent] config_table.md written to {root / 'config_table.md'}")
        export_config_examples(dest_dir=root)
    except Exception as exc:  # noqa: BLE001
        print(f"[pageboygent] Warning: could not generate config files: {exc}")


class PostInstall(install):
    def run(self) -> None:
        super().run()
        _export_config_docs()


class PostDevelop(develop):
    def run(self) -> None:
        super().run()
        _export_config_docs()


BASE_DIR = Path(__file__).parent
README = (BASE_DIR / "readme.MD").read_text(encoding="utf-8") if (BASE_DIR / "readme.MD").exists() else ""

setup(
    name="pageboygent",
    version="0.1.0",
    description="Pagebodygent can Arranging, Gathering and Exploring literature, help you Better Organize your Growing Expository Notes and Track research trends.",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Zircon Shawn",
    python_requires=">=3.13",
    package_dir={"": "src"},
    packages=find_packages(where="src", exclude=["tests", "tests.*"]),
    include_package_data=True,
    package_data={"Pageboygent": ["templates/admin/*.html", "tags.txt"]},
    install_requires=[
        "openai",
        "peewee",
        "requests",
        "pyzotero>=1.15.0",
        "markdown",
        "markdownify",
        "Flask",
        "Flask-Admin",
        "chromadb",
        "whoosh",
        "feedparser",
        "numpy",
        "ollama",
        "worthy",
    ],
    entry_points={
        "console_scripts": [
            "pageboygent=Pageboygent.cli:main",
            "pageboygent-export-config-docs=Pageboygent.config:_cli_export_config_docs",
        ],
    },
    cmdclass={
        "install": PostInstall,
        "develop": PostDevelop,
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: GPL-3.0 License",
        "Operating System :: OS Independent",
    ],
)