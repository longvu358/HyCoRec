# HyCoRec Setup & Data Download

The `README.md` describes the required steps but glosses over what actually happens
under the hood and what has to be fetched by hand. This doc spells out the concrete
commands, in order, and what each one produces.

All commands below assume you are in the repo root (`/home/longvh/projects/HyCoRec`)
unless noted otherwise. `uv` auto-discovers `pyproject.toml` in parent directories, so
running from `HyCoRec/` also works.

## 1. Install dependencies

```bash
uv venv --python 3.12
UV_HTTP_TIMEOUT=300 uv sync
```

`torch` pulls in several hundred MB of NVIDIA CUDA wheels (`nvidia-cudnn-cu13`, etc.).
The default 30s HTTP timeout is too short for these on a slow connection and `uv sync`
fails with `Failed to download distribution due to network timeout`. Raising
`UV_HTTP_TIMEOUT` (seconds) fixes it. Set it permanently if this keeps happening:

```bash
echo 'export UV_HTTP_TIMEOUT=300' >> ~/.zshrc
```

## 2. Download the raw dataset (automatic, scripted)

`crslab`'s `BaseDataset.__init__` (see `HyCoRec/crslab/data/dataset/base.py:44`)
downloads a resource zip the first time a dataset is constructed, verifies its SHA256,
and unpacks it — this is what the README calls "automatically download". It normally
fires when you run `run_crslab.py`, but that script goes on to build the model and
crashes with `FileNotFoundError` if the edger (step 3) hasn't been built yet. Use
`HyCoRec/download_dataset.py` instead to run just the download step:

```bash
cd HyCoRec
uv run download_dataset.py -d redial       # -> data/dataset/redial/nltk/
uv run download_dataset.py -d tgredial     # -> data/dataset/tgredial/pkuseg/
uv run download_dataset.py -d opendialkg   # -> data/dataset/opendialkg/nltk/
uv run download_dataset.py -d durecdial    # -> data/dataset/durecdial/jieba/
```

Only run the dataset(s) matching the config you plan to use
(`config/crs/hycorec/<dataset>.yaml`). Each is 100–200+ MB and hosted on a PKU
SharePoint share; a `.built` marker is written on success so re-running is a no-op.

For `redial`, this produces (among others):
`data/dataset/redial/nltk/{train,valid,test}_data.json`, `entity2id.json`,
`token2id.json`, `dbpedia_subkg.json`, `conceptnet_subkg.txt`, `word2vec.npy`.

## 3. Get ConceptNet side data (manual — not auto-downloaded)

`HyCoRec/edger/redial.py` (and `opendialkg.py`) additionally need
`data/conceptnet/en_side.txt`; `tgredial.py`/`durecdial.py` need
`data/conceptnet/zh_side.txt`. **Nothing in the codebase downloads these** — they are
not part of the crslab resource zips above and there is no URL for them anywhere in
`crslab/`. The only source is the bundled archive linked in `README.md`:

<https://pan.quark.cn/s/7ccc30301942>

This is a Quark Netdisk share that requires the web UI (or app) to save/download —
it can't be fetched with a plain `curl`/`wget` one-liner. Steps:

1. Open the link, save the share to your own Quark Netdisk, then download it.
2. Extract it and copy the `conceptnet/` folder into `HyCoRec/data/conceptnet/`, so you
   have `HyCoRec/data/conceptnet/en_side.txt` (and `zh_side.txt` if you need
   tgredial/durecdial).
3. That same archive also contains prebuilt `item_edger.pkl` / `entity_edger.pkl` /
   `word_edger.pkl` files — if present, you can skip step 4 and place them directly in
   `HyCoRec/data/edger/<dataset>/`.

## 4. Build the item/entity/word edger

Requires step 2 (dataset) and step 3 (`data/conceptnet/`) to be done first for the
target dataset:

```bash
cd HyCoRec
uv run run_edger.py -d redial
```

Writes `data/edger/redial/{item,entity,word}_edger.pkl`
(`HyCoRec/crslab/model/crs/hycorec/hycorec.py:172` loads these at model build time).

## 5. Run

```bash
cd HyCoRec
uv run run_crslab.py -c config/crs/hycorec/redial.yaml -g 0 -s 3407
```

Use `-g -1` to force CPU if you don't have a CUDA GPU available.

## Expected directory layout (redial)

```
HyCoRec/data/
├── conceptnet/
│   └── en_side.txt              # from Quark archive (step 3)
├── dataset/redial/nltk/         # from download_dataset.py (step 2)
│   ├── train_data.json
│   ├── valid_data.json
│   ├── test_data.json
│   ├── entity2id.json
│   ├── token2id.json
│   ├── dbpedia_subkg.json
│   ├── conceptnet_subkg.txt
│   └── word2vec.npy
└── edger/redial/                # from run_edger.py (step 4)
    ├── item_edger.pkl
    ├── entity_edger.pkl
    └── word_edger.pkl
```
