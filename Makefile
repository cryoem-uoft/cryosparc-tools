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

/usr/local/bin/micromamba:
	curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /usr/local bin/micromamba

.venv:
	micromamba create -p ./.venv -y -c conda-forge python=3.10 pip wheel cython numpy jupyter-book autodocsumm
	micromamba run -p ./.venv pip install -e ".[build]"

.vercel/output/config.json:
	mkdir -p .vercel/output
	echo '{"version":3,"cache":["/usr/local/bin/micromamba", ".venv/**","build/**","docs/_build/**"]}' > .vercel/output/config.json

vercelinstall: /usr/local/bin/micromamba .venv
	echo "Install complete"

vercelbuild: .vercel/output/config.json .venv
	micromamba run -p ./.venv jupyter-book build docs
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
	rm -rf .venv

.PHONY: clean all vercelinstall vercelbuild
