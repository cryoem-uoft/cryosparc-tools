# -----------------------------------------------------------------------------
#    Docs
# -----------------------------------------------------------------------------
docs:
	# API documentation is generated from the `cryosparc/api.pyi` type stubs file,
	# but sphinx expects a `.py` file.
	mv cryosparc/api.py cryosparc/api.py.bak
	mv cryosparc/api.pyi cryosparc/api.py
	-jupyter-book build docs
	mv cryosparc/api.py cryosparc/api.pyi
	mv cryosparc/api.py.bak cryosparc/api.py


# -----------------------------------------------------------------------------
#    THE FOLLOWING TARGETS ARE FOR CRYOSPARC/DEPLOYMENT USE ONLY
#    Do not use this Makefile for tools development.
#    Use uv instead. Refer to README.md
# -----------------------------------------------------------------------------

PY_EXT_SUFFIX=$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
TARGET=cryosparc/dataset/core$(PY_EXT_SUFFIX)

all: $(TARGET)

# -----------------------------------------------------------------------------
#    Primary build target
# -----------------------------------------------------------------------------

$(TARGET): cryosparc/include/cryosparc-tools/*.h cryosparc/dataset/dataset.c cryosparc/dataset/*.pyx cryosparc/dataset/*.pxd pyproject.toml
	pip install --no-build-isolation -Ceditable.mode=inplace -e .

# -----------------------------------------------------------------------------
#    Vercel deployment-related targets
# -----------------------------------------------------------------------------

/usr/local/bin/uv:
	curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/local/bin" sh

.venv:
	uv venv
	uv sync --group docs --no-group dev

.vercel/output/config.json:
	mkdir -p .vercel/output
	echo '{"version":3,"cache":["/usr/local/bin/uv", "/usr/local/bin/uvx", ".venv/**","build/**","docs/_build/**"]}' > .vercel/output/config.json

vercelinstall: /usr/local/bin/uv .venv
	echo "Install complete"

vercelbuild: .vercel/output/config.json .venv
	uv run --group docs --no-group dev make docs
	rm -rf .vercel/output/static && cp -R docs/_build/html .vercel/output/static
	echo "Build complete"

# -----------------------------------------------------------------------------
#    Cleanup
# -----------------------------------------------------------------------------

clean:
	rm -f $(TARGET)
	rm -f *.tgz *.tar.gz *.whl
	rm -f cryosparc/*.so
	rm -f cryosparc/dataset/*.so
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info
	rm -rf .venv
	rm -rf CMakeFiles CMakeInit.txt CMakeCache.txt cmake_install.cmake
	rm -rf cryosparc/dataset/CMakeFiles cryosparc/dataset/CMakeInit.txt \
		cryosparc/dataset/CMakeCache.txt cryosparc/dataset/cmake_install.cmake
	rm -rf build.ninja .ninja_deps .ninja_log

.PHONY: clean all docs vercelinstall vercelbuild
