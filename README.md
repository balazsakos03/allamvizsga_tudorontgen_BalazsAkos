# 🩻 Tüdőröntgen felvételek osztályozása mély tanulású neurális hálózatokkal

Ez a repository a szakdolgozatom megvalósítását tartalmazza:  
**„Tüdőröntgen felvételek diagnosztikai célú osztályozása mély tanulású neurális hálózatokkal”**

---

## 📌 Projekt leírása
A projekt célja, hogy mélytanulási neurális hálózatok segítségével  
**mellkasröntgen felvételeket automatikusan osztályozzon**,  
és összehasonlítsa különböző modellek és tanítási stratégiák teljesítményét.

A hangsúly nem egyetlen „végső” modell megalkotásán van,  
hanem a **modellek fejlődési folyamatának**,  
valamint azok **eredményeinek és tapasztalatainak dokumentálásán**.

---

## 🧪 Kísérleti megközelítés
A projekt során:
- kezdeti, egyszerű baseline modellek kerülnek megvalósításra,
- majd fokozatosan összetettebb, Transfer Learning alapú architektúrák,
- amelyek teljesítménye különböző metrikák segítségével kerül összehasonlításra.

Az egyes kísérletek és eredmények részletes leírása a  
`results/README.md` fájlban található.

---

## 📂 Repository felépítése
├── data/ # Felhasznált adatkészletek dokumentációja
├── notebooks/ # Tanítási és kiértékelési notebookok
├── results/ # Modellek eredményei, metrikák és grafikonok
├── src/ # Közös forráskód (modellek, segédfüggvények)
├── .gitignore
└── README.md


- Az adatkészletek **nem kerülnek feltöltésre**, csak dokumentálásra  
  → lásd: `data/README.md`
- A modellekhez tartozó részletes tapasztalatok  
  → lásd: `results/README.md`

---

## 🛠️ Felhasznált technológiák
- Python 3.10+  
- TensorFlow / Keras  
- PyTorch  
- NumPy, OpenCV, Matplotlib  
- **Kaggle Notebook környezet** (fő futtatási platform)

A projekt során több neurális hálózat és mélytanulási megközelítés kerül kipróbálásra,  
különös tekintettel a Transfer Learning technikákra és az architektúrák összehasonlítására.

---

## 👨‍💻 Szerző
- **Balázs Ákos**  
Szakdolgozat – 2025