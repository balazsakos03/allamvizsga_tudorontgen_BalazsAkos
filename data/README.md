# 📂 Datasets – áttekintés

Ez a mappa kizárólag **dokumentációs célokat** szolgál.  
A projekthez használt adatkészletek **nem kerülnek feltöltésre a repository-ba**,  
hanem külső forrásból (Kaggle) kerülnek felhasználásra.

---

## 🩻 Chest X-Ray Images (Pneumonia)

- Forrás: Kaggle  
- Dataset: *Chest X-Ray Images (Pneumonia)*  
- Link: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia  

### Felhasználás
Ez az adatkészlet az alábbi modellek tanításához és kiértékeléséhez lett felhasználva:

- **02_transfer_learning_resnet.ipynb**  
  - Modell: ResNet50  
  - Feladat: bináris osztályozás  
    - NORMAL  
    - PNEUMONIA  

A dataset a Kaggle Notebook környezetben Input Datasetként került csatolásra,  
és az alábbi elérési úton lett használva:

`/kaggle/input/chest-xray-pneumonia/chest_xray/`


---

## ℹ️ Megjegyzés
A pneumonia osztályon belül a fájlnevek tartalmaznak információt a betegség eredetéről  
(bakteriális / vírusos), azonban ez a modell **bináris osztályozásként** kezeli az adatokat.  
A multiclass feldolgozás külön kísérletként kerül megvalósításra.