"""CheXpert Plus slice for study 2 (docs/RESEARCH_PIVOT.md §9): download only the frontal adult images we need, downsized.

Source: Stanford AIMI on Redivis, dataset aimi.chexpert_plus:5yyj (v1.0). Needs the user's approved data access. Auth is the
redivis client's own sign-in (OAuth on first use, cached by the library) or REDIVIS_API_TOKEN set by the user; this script
never reads, prints or stores credentials.

    .venv/bin/python scripts/chexpert/fetch.py probe     # sign-in check: first PNG file names (also triggers the OAuth prompt)
    .venv/bin/python scripts/chexpert/fetch.py meta      # metadata table (no report text) + CheXbert label files
    .venv/bin/python scripts/chexpert/fetch.py select    # frontal, adult, opacity-type vs No Finding -> selection.csv
    .venv/bin/python scripts/chexpert/fetch.py images    # download the selected PNGs, longer side -> 512 px (resumable)
    .venv/bin/python scripts/chexpert/fetch.py select-gen          # larger GENERATOR set -> selection_gen.csv
    .venv/bin/python scripts/chexpert/fetch.py select-gen obvious  # same, positives = Consolidation / Pneumonia -> selection_obv.csv
    .venv/bin/python scripts/chexpert/fetch.py images selection_gen.csv
Output: data/chexpert/{meta.parquet, labels/*.json, selection.csv, selection_gen.csv, images/<path>.png}
select-gen (docs/RESEARCH_PIVOT.md §13, more data): every frontal adult No Finding image of a CheXpert train patient that is
not in our val / test splits (all images per patient, ~14.8k) + as many opacity-type images (random, of ~97.8k); same
label rule as `select`. Only the generator sees these; the classifier manifests are unchanged.
"""
import io
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/chexpert"
DS = "aimi.chexpert_plus:5yyj:v1_0"
META_TABLE, LABEL_TABLE, PNG_TABLE = "df_chexpert_plus_240401:bavj", "chexpert_labels:pmec", "png_train:s6cj"
META_COLS = ["path_to_image", "frontal_lateral", "ap_pa", "deid_patient_id", "patient_report_date_order", "age", "sex",
             "split"]
POSITIVE = ["Lung Opacity", "Pneumonia", "Consolidation"]
PER_CLASS = 6000  # enough for a ~5k-image generator training set plus classifier pools; one study per patient
SIDE = 512


def table(name):
    import redivis
    return redivis.table(f"{DS}.{name}")


def probe():
    for f in table(PNG_TABLE).list_files(max_results=5):
        print(f.properties["file_name"], f.properties["size"])


def meta():
    OUT.mkdir(parents=True, exist_ok=True)
    df = table(META_TABLE).to_pandas_dataframe(variables=META_COLS)
    df.to_parquet(OUT / "meta.parquet")
    print("meta", df.shape, df.frontal_lateral.value_counts().to_dict(), df.ap_pa.value_counts().to_dict())
    (OUT / "labels").mkdir(exist_ok=True)
    for f in table(LABEL_TABLE).list_files():
        f.download(str(OUT / "labels") + "/")
        print("labels", f.properties["file_name"], f.properties["size"] // 2**20, "MB")


def read_labels(path):
    import pandas as pd
    try:
        return pd.read_json(path, lines=True)
    except ValueError:
        return pd.read_json(path)


def select():
    import numpy as np
    import pandas as pd
    m = pd.read_parquet(OUT / "meta.parquet")
    lab = read_labels(OUT / "labels/impression_fixed.json")  # CheXbert on the impression section
    print("label columns:", list(lab.columns)[:20])
    key = "path_to_image" if "path_to_image" in lab else lab.columns[0]
    d = m.merge(lab, left_on="path_to_image", right_on=key, how="inner")
    d = d[(d.frontal_lateral.str.lower() == "frontal") & (pd.to_numeric(d.age, errors="coerce") >= 18)]
    pos = (d[["Consolidation", "Pneumonia"] if kind == "obvious" else POSITIVE] == 1).any(axis=1)
    neg = (d["No Finding"] == 1) & ~(d[POSITIVE].isin([1, -1])).any(axis=1)
    d = d[pos | neg].assign(label=pos.astype(int))
    # one image per patient (earliest study), so patient-level splits stay simple
    d = d.sort_values(["deid_patient_id", "patient_report_date_order"]).drop_duplicates("deid_patient_id")
    rng = np.random.default_rng(0)
    sel = pd.concat([g.iloc[rng.permutation(len(g))[:PER_CLASS]] for _, g in d.groupby("label")])
    sel.to_csv(OUT / "selection.csv", index=False)
    print("selected", sel.label.value_counts().to_dict(), "AP/PA", sel.groupby("label").ap_pa.value_counts().to_dict())


def select_gen(kind="mixed"):  # "obvious": positives = Consolidation or Pneumonia (docs/RESEARCH_PIVOT.md §13) -> selection_obv.csv
    import numpy as np
    import pandas as pd
    m = pd.read_parquet(OUT / "meta.parquet")
    lab = read_labels(OUT / "labels/impression_fixed.json")
    key = "path_to_image" if "path_to_image" in lab else lab.columns[0]
    d = m.merge(lab, left_on="path_to_image", right_on=key, how="inner")
    d = d[(d.frontal_lateral.str.lower() == "frontal") & (pd.to_numeric(d.age, errors="coerce") >= 18) & (d.split == "train")]
    man = pd.read_csv(ROOT / "artifacts/manifests/chexpert.csv", dtype={"patient_id": str})
    d = d[~d.deid_patient_id.isin(set(man[man.split != "train"].patient_id))]
    pos = (d[["Consolidation", "Pneumonia"] if kind == "obvious" else POSITIVE] == 1).any(axis=1)
    neg = (d["No Finding"] == 1) & ~(d[POSITIVE].isin([1, -1])).any(axis=1)
    d = d[pos | neg].assign(label=pos.astype(int))
    n = int(min((d.label == 0).sum(), d.label.sum()))
    rng = np.random.default_rng(0)
    pick = {y: d[d.label == y].iloc[rng.permutation(int((d.label == y).sum()))[:n]] for y in (1, 0)}  # positives first: same draw as before
    sel = pd.concat([pick[0], pick[1]])
    sel.to_csv(OUT / ("selection_obv.csv" if kind == "obvious" else "selection_gen.csv"), index=False)
    print("generator selection", sel.label.value_counts().to_dict(), "patients", sel.deid_patient_id.nunique())


def images(selection="selection.csv"):
    import pandas as pd
    from PIL import Image
    sel = pd.read_csv(OUT / selection)
    tbl = table(PNG_TABLE)
    want = {Path(p).with_suffix(".png").as_posix().removeprefix("train/") for p in sel.path_to_image}  # PNG_train names

    todo = [n for n in sorted(want) if not (OUT / "images" / n).exists()]
    print(f"listing files ({len(todo)} of {len(want)} still to fetch)", flush=True)
    files = {f.properties["file_name"]: f for f in tbl.list_files()} if todo else {}  # one listing, not one lookup per file
    missing = [n for n in todo if n not in files]
    print(f"listed {len(files)} files; {len(missing)} selected names not found, e.g. {missing[:3]}", flush=True)

    def fetch(name):
        dst = OUT / "images" / name
        if dst.exists() or name not in files:
            return 0
        img = Image.open(io.BytesIO(files[name].read())).convert("L")
        s = SIDE / max(img.size)
        img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".tmp.png")
        img.save(tmp)
        tmp.rename(dst)
        return 1

    names = todo
    done = 0
    with ThreadPoolExecutor(8) as ex:
        for i, r in enumerate(ex.map(fetch, names)):
            done += r
            if i % 500 == 0:
                print(f"{i}/{len(names)} ({done} new)", flush=True)
    print("images done", len(names))


if __name__ == "__main__":
    {"probe": probe, "meta": meta, "select": select, "select-gen": select_gen, "images": images}[sys.argv[1]](*sys.argv[2:])
