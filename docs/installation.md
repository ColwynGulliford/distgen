Installing distgen
===============

Installing `distgen` from the `conda-forge` channel can be achieved by adding `conda-forge` to your channels with:

```shell
conda config --add channels conda-forge
```

Once the `conda-forge` channel has been enabled, `distgen` can be installed with:

```shell
conda install distgen
```

It is possible to list all of the versions of `distgen` available on your platform with:

```shell
conda search distgen --channel conda-forge
```



Developers
==========


Clone this repository:
```shell
git clone https://github.com/ColwynGulliford/distgen.git
```

Create an environment `distgen-dev` with all the dependencies:
```shell
conda env create -f environment.yml
```


Install as editable:
```shell
conda activate distgen-dev
pip install --no-dependencies -e .
```



