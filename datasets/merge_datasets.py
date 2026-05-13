import os
import shutil
import random
from pathlib import Path

#config
BASE_DIR    = Path(__file__).parent
KAGGLE_DIR  = BASE_DIR / "chest_xray"
COVID_DIR   = BASE_DIR / "covid_radiography"
OUTPUT_DIR  = BASE_DIR / "combined_dataset"

SEED        = 42
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
# TEST_RATIO  = 0.10 (a maradek)

random.seed(SEED)

#config undersampling
#ha True, a Normal osztalyt lecsokkenjtuk a Pneumonia meretehez
#ha False, minden kepet hasznalunk (imbalanced dataset)
UNDERSAMPLE_NORMAL = True

def collect_images(directory: Path) -> list:
    extensions = {f.lower() for f in [".jpg", ".jpeg", ".png"]}
    return [str(p) for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in extensions]

def split_files(files: list, train_r: float, val_r: float) -> tuple:
    random.shuffle(files)
    n = len(files)
    train_end = int(n * train_r)
    val_end   = int(n * (train_r + val_r))
    return files[:train_end], files[train_end:val_end], files[val_end:]

def copy_files(files: list, dest_dir: Path, label: str):
    dest_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        filename = Path(src).name
        dest = dest_dir / filename
        if dest.exists():
            dest = dest_dir / f"{Path(src).parent.parent.name}_{filename}"
        shutil.copy2(src, dest)

def print_stats(label: str, train: list, val: list, test: list):
    print(f"  {label}: train={len(train)}, val={len(val)}, test={len(test)}, total={len(train)+len(val)+len(test)}")

#kepek osszegyujtese
print("=" * 60)
print("Collecting images...")
print("=" * 60)

#kaggle chest x-ray - mar fel van osztva train/val/test mappakba
kaggle_normal_all    = []
kaggle_pneumonia_all = []

for split in ["train", "val", "test"]:
    kaggle_normal_all    += collect_images(KAGGLE_DIR / split / "NORMAL")
    kaggle_pneumonia_all += collect_images(KAGGLE_DIR / split / "PNEUMONIA")

print(f"Kaggle - Normal:    {len(kaggle_normal_all)} kep")
print(f"Kaggle - Pneumonia: {len(kaggle_pneumonia_all)} kep")

#covid-19 radiography - flat struktura
covid_normal_all    = collect_images(COVID_DIR / "Normal" / "images")
covid_pneumonia_all = collect_images(COVID_DIR / "Viral Pneumonia" / "images")

print(f"COVID  - Normal:    {len(covid_normal_all)} kep")
print(f"COVID  - Pneumonia: {len(covid_pneumonia_all)} kep")

#osztalyok osszevonasa
all_normal    = kaggle_normal_all + covid_normal_all
all_pneumonia = kaggle_pneumonia_all + covid_pneumonia_all

print(f"\nOsszesitett - Normal:    {len(all_normal)} kep")
print(f"Osszesitett - Pneumonia: {len(all_pneumonia)} kep")

#undersampling
if UNDERSAMPLE_NORMAL and len(all_normal) > len(all_pneumonia):
    random.shuffle(all_normal)
    all_normal = all_normal[:len(all_pneumonia)]
    print(f"\nUndersampling utan - Normal: {len(all_normal)} kep")
    print(f"Undersampling utan - Pneumonia: {len(all_pneumonia)} kep")

#felosztas
print("\nSplitting into train/val/test (80/10/10)...")

normal_train,    normal_val,    normal_test    = split_files(all_normal,    TRAIN_RATIO, VAL_RATIO)
pneumonia_train, pneumonia_val, pneumonia_test = split_files(all_pneumonia, TRAIN_RATIO, VAL_RATIO)

print_stats("NORMAL",    normal_train,    normal_val,    normal_test)
print_stats("PNEUMONIA", pneumonia_train, pneumonia_val, pneumonia_test)

#output
print(f"\nCopying to: {OUTPUT_DIR}")
print("(Ez eltarthat egy-ket percig...)")

copy_files(normal_train,    OUTPUT_DIR / "train" / "NORMAL",    "NORMAL")
copy_files(normal_val,      OUTPUT_DIR / "val"   / "NORMAL",    "NORMAL")
copy_files(normal_test,     OUTPUT_DIR / "test"  / "NORMAL",    "NORMAL")
copy_files(pneumonia_train, OUTPUT_DIR / "train" / "PNEUMONIA", "PNEUMONIA")
copy_files(pneumonia_val,   OUTPUT_DIR / "val"   / "PNEUMONIA", "PNEUMONIA")
copy_files(pneumonia_test,  OUTPUT_DIR / "test"  / "PNEUMONIA", "PNEUMONIA")

#statisztika
print("\n" + "=" * 60)
print("COMBINED DATASET - KESZ")
print("=" * 60)

for split in ["train", "val", "test"]:
    for cls in ["NORMAL", "PNEUMONIA"]:
        n = len(list((OUTPUT_DIR / split / cls).glob("*")))
        print(f"  {split}/{cls}: {n} kep")

total = sum(
    len(list((OUTPUT_DIR / split / cls).glob("*")))
    for split in ["train", "val", "test"]
    for cls in ["NORMAL", "PNEUMONIA"]
)
print(f"\n  TOTAL: {total} kep")
print(f"  Output: {OUTPUT_DIR}")
print("=" * 60)