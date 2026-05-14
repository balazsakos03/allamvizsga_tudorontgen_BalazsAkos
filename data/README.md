# Datasets – áttekintés

Ez a mappa kizárólag dokumentációs célokat szolgál.
A projekthez használt adatkészletek nem kerülnek feltöltésre a repository-ba,
hanem külső forrásból (Kaggle) kerülnek felhasználásra, illetve lokálisan
kerülnek összeállításra a `merge_datasets.py` szkript segítségével.

---

## 1. Chest X-Ray Images (Pneumonia)

- **Forrás:** Kaggle — Paul Mooney
- **Link:** https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- **Osztályok:** NORMAL, PNEUMONIA (bináris)
- **Felosztás:** train / val / test (előre felosztva)
- **Méret:** ~5,216 kép

Ez az adatkészlet az első tíz kísérlet alapját képezi (notebook 01–10).
A Kaggle Notebook környezetben Input Datasetként került csatolásra,
az alábbi elérési úton:

`/kaggle/input/chest-xray-pneumonia/chest_xray/`

Megjegyzés: a PNEUMONIA osztályon belül a fájlnevek tartalmazzák a betegség eredetét
(bakteriális / vírusos), azonban a kísérletek ezt bináris feladatként kezelik.

---

## 2. COVID-19 Radiography Database

- **Forrás:** Kaggle — Tawsifur Rahman
- **Link:** https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
- **Osztályok:** Normal, Viral Pneumonia (a COVID és Lung Opacity osztályokat kihagytuk)
- **Felosztás:** nincs előre felosztva, 80/10/10 arányban kerül felosztásra
- **Méret:** ~10,192 Normal + ~1,345 Viral Pneumonia kép

Ez az adatkészlet a 11-es kísérletben (cross-dataset kiértékelés) került felhasználásra.
Felnőtt betegek felvételeit tartalmazza, homogénebb képminőséggel mint az első dataset.
A Kaggle Notebook környezetben az alábbi elérési úton érhető el:

`/kaggle/input/covid19-radiography-database/COVID-19_Radiography_Dataset/`

---

## 3. Kombinált adatkészlet (Combined Dataset)

- **Forrás:** az 1. és 2. adatkészlet összevonásával lokálisan összeállítva
- **Szkript:** `datasets/merge_datasets.py`
- **Osztályok:** NORMAL, PNEUMONIA (bináris)
- **Felosztás:** train / val / test (80/10/10)
- **Méret:** ~11,236 kép (undersamplinggal kiegyensúlyozva: 5,618 Normal + 5,618 Pneumonia)

Az összevonás célja a domain shift probléma csökkentése: a kombinált dataset
különböző forrásokból, különböző készülékekkel és különböző betegcsoportokon
(gyerek és felnőtt) készült felvételeket egyaránt tartalmaz, ezáltal
robusztusabb modell tanítható rajta.

Az összevonás lépései:
- mindkét datasetből összegyűjtésre kerültek a Normal és Pneumonia képek
- a Normal osztály undersamplinggal a Pneumonia osztály méretére lett csökkentve
- a képek 80/10/10 arányban kerültek felosztásra stratifikált mintavétellel

A kombinált adatkészlet Kaggle-re feltöltve saját datasetként érhető el,
az alábbi elérési úton:

`/kaggle/input/combined-chest-xray-dataset/combined_dataset/`

---

## Megjegyzés a reprodukálhatósághoz

A nyers datasetek nem kerülnek feltöltésre a repository-ba (lásd `.gitignore`).
A kombinált dataset reprodukálásához:

1. Töltsd le az 1. és 2. datasetet a fenti Kaggle linkekről
2. Csomagold ki a `datasets/chest_xray/` és `datasets/covid_radiography/` mappákba
3. Futtasd: `python datasets/merge_datasets.py`