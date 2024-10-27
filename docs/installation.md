# Installing Distgen


##  Using conda


```bash
conda install -c conda-forge pytao
```

## Using setuptools

```bash
python setup.py install
```



## Developers


Clone this repository:
```shell
git clone https://github.com/ColwynGulliford/distgen.git
```

Create an environment `distgen-dev` with all the dependencies:
```shell
conda env create -f environment-dev.yml
```

Install as editable:
```shell
conda activate distgen-dev
pip install --no-dependencies -e .
```

Create documentation:
```shell
mkdocs serve
```
