# NRTK EXPLORER

NRTK Explorer is a web application for exploring image datasets. It provides
insights of a image dataset in [COCO][3] format and it evaluate image
transformation and perturbation resilience of object recognition DL models. It
is built using [trame][1] by the [kitware][2] team.

![nrtk explorer screenshot](https://github.com/user-attachments/assets/85c95836-3490-40ec-813d-e6841c540d51)

## Features

- Explore image datasets in COCO format.
- Apply parametrized image degradation (such as blur) to the images.
- Benchmark dataset resilience with a differential PCA|UMAP analysis on the
  embeddings of the images and their transformations.
- Evaluate object detection DL models in both the source images and its
  transformations.
- When possible it will attempt to utilize the user GPU as much as possible to
  speedup its computations.

## Installing

Install it from pypi:

```bash
pip install nrtk-explorer
```

## Usage

Explore Hugging Face hosted [dataset](https://huggingface.co/datasets/rafaelpadilla/coco2017):

```bash
nrtk-explorer --dataset rafaelpadilla/coco2017
```

Compare inference results for Hugging Face hosted models:

```bash
nrtk-explorer --dataset cppe-5 --models qubvel-hf/detr-resnet-50-finetuned-10k-cppe5 ashaduzzaman/detr_finetuned_cppe5
```

2 COCO format datasets are available at: https://github.com/vicentebolea/nrtk_explorer_datasets/

```bash
git clone https://github.com/vicentebolea/nrtk_explorer_datasets.git
nrtk-explorer --dataset ./nrtk_explorer_datasets/coco-od-2017/mini_val2017.json ./nrtk_explorer_datasets/OIRDS_v1_0/oirds.json
```

## CLI flags and options

- `--dataset` specify the path to a [COCO dataset](https://roboflow.com/formats/coco-json) JSON file,
  a [Hugging Face dataset](https://huggingface.co/datasets?task_categories=task_categories:object-detection) repository name,
  or a directory loadable by the [Dataset](https://huggingface.co/docs/datasets/index) library.
  You can specify multiple datasets using a space as the
  separator. Example: `nrtk-explorer --dataset ../foo-dir/coco.json cppe-5`
- `--repository` Specify an existing directory where exported datasets will be saved to and loaded from.
- `--download` Cache Hugging Face Hub datasets locally instead of streaming them.
  When datasets are streamed, nrtk-explorer limits the number of loaded images.
- `--models` specify the Hugging Face Hub [object detection](https://huggingface.co/models?pipeline_tag=object-detection&library=transformers&sort=trending)
  repository name or a directory loadable by the [Transformers](https://huggingface.co/docs/transformers/index) library. Load multiple models using space as the separator.  
  Example: `nrtk-explorer --models hustvl/yolos-tiny facebook/detr-resnet-50`
- `-h|--help` show the help for the command line options. nrtk-explorer inherits the trame
  command line options and flags.

![nrtk explorer usage](https://github.com/user-attachments/assets/86a61485-471c-4b94-872e-943cb9da52a1)

## Contribute to NRTK_EXPLORER

```bash
git clone https://github.com/Kitware/nrtk-explorer.git
cd nrtk-explorer
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e '.[dev]'
pytest .
```

For more details on setting up a development environment see [DEVELOPMENT docs](docs/source/manual/DEVELOPMENT.rst).

### Create release

1. Merge `main` to `release` with a _merge commit_.
2. Run "Create Release" workflow with workflow from `release` branch.
3. Merge `release` to `main` with a _merge commit_.
4. Check package versions in Conda Feedstock [meta.yaml file](https://github.com/conda-forge/nrtk-explorer-feedstock/blob/main/recipe/meta.yaml)

[1]: https://trame.readthedocs.io/en/latest/
[2]: https://www.kitware.com/
[3]: https://cocodataset.org/
