TARGET=cryosparc/core.cpython-310-x86_64-linux-gnu.so  # for CryoSPARC

all: $(TARGET)

# -----------------------------------------------------------------------------
#    Primary build target
# -----------------------------------------------------------------------------

$(TARGET): cryosparc/include/cryosparc-tools/*.h cryosparc/dataset.c cryosparc/*.pyx cryosparc/*.pxd setup.py pyproject.toml
	python -m setup build_ext -i

# -----------------------------------------------------------------------------
#    Vercel deployment-related targets
# -----------------------------------------------------------------------------

.venv/bin/python:
	python3 -m venv .venv

.venv/bin/pip: .venv/bin/python
	.venv/bin/python -m pip install -U pip wheel

.venv/bin/jupyter-book: .venv/bin/pip
	.venv/bin/pip install -e ".[build]"

.vercel/output/config.json:
	mkdir -p .vercel/output
	echo '{"version":3,"cache":[".venv/**","build/**","docs/_build/**","Python-*.tgz","Python-*/**"]}' > .vercel/output/config.json

verceldeps:
	yum update -y
	yum install bzip2-devel libffi-devel openssl-devel sqlite-devel -y

vercelinstall: verceldeps python .venv/bin/python
	echo "Install complete"

vercelbuild: .vercel/output/config.json .venv/bin/jupyter-book
	.venv/bin/jupyter-book build docs
	rm -rf .vercel/output/static && cp -R docs/_build/html .vercel/output/static

# -----------------------------------------------------------------------------
#    Cleanup
# -----------------------------------------------------------------------------

clean:
	rm -f $(TARGET)
	rm -f *.tgz *.tar.gz
	rm -f cryosparc/*.so
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info
	rm -rf Python-$(PYTHON_VERSION)

.PHONY: clean all python verceldeps vercelinstall vercelbuild
