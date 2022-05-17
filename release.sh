#!/usr/bin/env bash

# hacky helper script to draft a new release
# ------------------------------------------
#
# requires (and checks) that local repo is in branch 'main' and in the same state as 'origin/main'
# performs some sanity checks, before drafting a new release on github
# after release is manually published on github, package is automatically upload to pypi via github action
#
# notes:
# - only tested on mac
# - needs git and github cli (https://github.com/cli/cli) with proper credentials for repo

# go to folder where this script lives
root="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $root

[[ `git branch --show-current` == "main" ]] || { echo ">>> not on main branch, exiting."; exit 1; }

# https://stackoverflow.com/questions/3258243/check-if-pull-needed-in-git#comment20583319_12791408
[[ `git rev-list HEAD...origin/main --count` == 0 ]] || { echo ">>> repo not in same state as origin/main, exiting,"; exit 1; }

[[ `git status --porcelain --untracked-files=no` ]] && { echo ">>> there are local changes, exiting."; exit 1; }

eval $(
python -W ignore - << END
import stardist_napari, pathlib
print(f'path={pathlib.Path(stardist_napari.__path__[0]).parent}')
print(f'version={stardist_napari.__version__}')
END
)

[[ $path == $root ]] || { echo ">>> not in package directory, exiting."; exit 1; }

date_str=`date +"%Y.%-m.%-d"`
[[ $version == $date_str ]] || { echo ">>> package version (${version}) doesn't match current date (${date_str}), exiting."; exit 1; }

gh release view $version > /dev/null 2>&1
[[ $? == 1 ]] || { gh release list --limit 5; echo ">>> release $version already exists, exiting."; exit 1; }

gh release create $version --title $version --draft --target main
