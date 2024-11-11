# Syntactic disambiguation as a test of visual-textual integration in autoregressive visual language models

1. Tune the packages to your needs by editing `requirements.txt` as desired.
    - Note that some packages must be installed with conda rather than pip. Please place the full install command in `requirements-conda.sh` (i.e., `conda install -yc <channel> <package>==<version>`).
2. Use `make env` to create your environment!
    - Note that this expects you to already be in the conda base environment.
3. After writing some code, make it conform to style guides with `make format`.
    - If you added more packages, don't forget to add them to `requirements.txt` or `requirements-conda.sh`! I also recommend then running (in base) `make uninstall` and `make env` to ensure your experience is still replicable for others.
4. Before pushing, check that there are no type errors and that the linter passes with `make test`.
5. After you complete your module, use `make run` to run the package with default arguments as specified in `args.py`. Use `python -m <package> [args]` for a more flexible interface.
6. If you forget any of these commands, use `make help` or simply view the Makefile.
