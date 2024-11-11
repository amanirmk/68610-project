# Lullaby: Preventing Code Management Nightmares
This is a template meant to encourage higher code quality and allow for more seamless collaboration between multiple contributors. Know that not all code is my own, in fact, most has been borrowed from [Ben Lipkin](https://benlipkin.github.io).

1. Get started by replacing all of the instances of "lullaby" in the repository, including in `Makefile` and `setup.py`.
    - Do *not* make changes to `main-guard.json` should you choose to use it.
    - Tip: With some code editors, changing the folder name will prompt an automatic fix to the dependencies inside.
2. Tune the packages to your needs by editing `requirements.txt` as desired.
    - Note that some packages must be installed with conda rather than pip. Please place the full install command in `requirements-conda.sh` (i.e., `conda install -yc <channel> <package>==<version>`).
3. Use `make env` to create your environment!
    - Note that this expects you to already be in the conda base environment.
4. After writing some code, make it conform to style guides with `make format`.
    - If you added more packages, don't forget to add them to `requirements.txt` or `requirements-conda.sh`! I also recommend then running (in base) `make uninstall` and `make env` to ensure your experience is still replicable for others.
5. Before pushing, check that there are no type errors and that the linter passes with `make test`.
6. After you complete your module, use `make run` to run the package with default arguments as specified in `args.py`. Use `python -m <package> [args]` for a more flexible interface.
7. If you forget any of these commands, use `make help` or simply view the Makefile.
8. Lastly, protect yourself from future headaches by adding the `main-guard.json` ruleset to your repository! This will require everyone to create a pull request that passes `make test` before merging with the main branch. After you do, feel free to delete `main-guard.json` from the template. If you don't use it, then delete it along with `.github/`.
