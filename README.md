# Sample PyPI package + GitHub Actions + Versioneer

This template aims to automate the tedious and error-prone steps of tagging/versioning, building and publishing new package versions. This is achieved by syncing git tags and versions with Versioneer, and automating the build and release with GitHub Actions, so that publishing a new version is as painless as:

```console
$ git tag vX.Y.Z && git push --tags
```

The following guide assumes familiarity with `setuptools` and PyPI. For an introduction to Python packaging, see the references at the bottom.

## How to use this template

1. Click on the *Use this template* button to get a copy of this repository.
1. Rename *src/sample_package* folder to your package name &mdash; *src/* is where your package must reside.
1. Go through each of the following files and rename all instances of *sample-package* or *sample_package* to your package name. Also update the package information such as author names, URLs, etc.
    1. setup.py
    1. pyproject.toml
    1. \_\_init\_\_.py
1. Install `versioneer` and `tomli`, and run `versioneer`:

    ```console
    $ pip install tomli
    $ pip install versioneer
    $ versioneer install
    ```
    Then *commit* the changes produced by `versioneer`. See [here](https://github.com/python-versioneer/python-versioneer/blob/master/INSTALL.md) to learn more.
1. Setup your PyPI credentials. See the section *Saving credentials on Github* of [this guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/). You should use the variable names `TEST_PYPI_API_TOKEN` and `PYPI_API_TOKEN` for the TestPyPI and PyPI tokens, respectively. See *.github/workflows/release.yaml*.

You are all set! It should now be possible to run `git tag vX.Y.Z && git push --tags` to automatically version, build and publish a new release to PyPI.

Finally, it is a good idea to [configure tag protection rules](https://docs.github.com/en/enterprise-server@3.8/repositories/managing-your-repositorys-settings-and-features/managing-repository-settings/configuring-tag-protection-rules) in your repository.

## References
- Python packaging guide: https://packaging.python.org/en/latest/tutorials/packaging-projects/
    - ...and how things are changing: https://snarky.ca/what-the-heck-is-pyproject-toml/ &mdash; in particular, note that while *pyproject.toml* seems to be the future, currently Versioneer still depends on *setup.py*.
- Versioneer: https://github.com/python-versioneer/python-versioneer
- GitHub Actions: https://docs.github.com/en/actions
