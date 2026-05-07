# Datasets – áttekintés

Ez a mappa kizárólag dokumentációs célokat szolgál.
A projekthez használt adatkészletek nem kerülnek feltöltésre a repository-ba,
hanem külső forrásból (Kaggle) kerülnek felhasználásra.

---

## Chest X-Ray Images (Pneumonia)

- **Forrás:** Kaggle — Paul Mooney
- **Link:** https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- **Osztályok:** NORMAL, PNEUMONIA (bináris)
- **Felosztás:** train / val / test

Ez az adatkészlet az összes eddigi kísérlet alapját képezi (notebook 01–10).
A Kaggle Notebook környezetben Input Datasetként került csatolásra,
az alábbi elérési úton:

`/kaggle/input/chest-xray-pneumonia/chest_xray/`

Megjegyzés: a PNEUMONIA osztályon belül a fájlnevek tartalmazzák a betegség eredetét
(bakteriális / vírusos), azonban a jelenlegi kísérletek ezt bináris feladatként kezelik.
Multiclass osztályozás külön kísérletként kerül megvalósításra.

---

## Második adatkészlet (cross-dataset kiértékeléshez)

A cross-dataset generalizáció vizsgálatához egy második, független forrásból származó
adatkészlet kerül felhasználásra. Ez lehetővé teszi annak mérését, hogy a modellek
nem csupán a saját teszthalmazukon, hanem eltérő körülmények között készült felvételeken
is megbízhatóan teljesítenek-e.

A konkrét adatkészlet és elérési útja a kísérlet megkezdésekor kerül dokumentálásra.