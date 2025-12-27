# 📊 Kísérleti eredmények és modellek

Ez a mappa a projekt során megvalósított neurális hálózatok  
**eredményeit, kiértékeléseit és tapasztalatait** tartalmazza.

A cél nem egyetlen modell bemutatása, hanem a **modellek fejlődési folyamatának**  
és összehasonlításának dokumentálása.

---

## 1️⃣ Kezdeti baseline modell

**Notebook:**  
`notebooks/01_baseline_cnn.ipynb`

### Modell leírása
Az első modell egy **kezdetleges, baseline jellegű konvolúciós neurális háló**,  
amelynek célja elsősorban:

- a feldolgozási pipeline kipróbálása,
- az adatkészlet alapvető megértése,
- valamint egy referencia pont létrehozása a későbbi modellekhez.

A modell:
- egyszerű CNN architektúrát alkalmazott,
- nem használt előtanított súlyokat,
- nem tartalmazott finomhangolást vagy komplex regularizációt.

### Megjegyzés
Ez a modell **nem tekinthető végleges megoldásnak**,  
hanem egy **tanulási és validációs lépés** volt a projekt elején.  
Az itt elért eredmények elsősorban összehasonlítási alapként szolgálnak.

---

## 2️⃣ Transfer Learning alapú referencia modell – ResNet50

**Notebook:**  
`notebooks/02_transfer_learning_resnet.ipynb`

### Modell leírása
A második kísérlet egy **Transfer Learning alapú konvolúciós neurális háló**,  
amely a ResNet50 architektúrát használja ImageNet-en előtanított súlyokkal.

A modell:
- bináris osztályozási feladatot old meg (NORMAL vs. PNEUMONIA),
- adataugmentációt alkalmaz a tanítóhalmazon,
- kezeli az osztályegyensúlytalanságot,
- kétfázisú tanítást használ:
  - fagyasztott backbone
  - finomhangolt felső rétegek

### Kiértékelés
A modell teljesítménye a teszt halmazon került kiértékelésre az alábbi metrikákkal:
- accuracy
- precision, recall, F1-score
- confusion matrix
- ROC-görbe és AUC érték

Az eredményeket a következő mappák tartalmazzák:
- `results/figures/` – grafikus kiértékelések
- `results/metrics/` – numerikus metrikák (JSON)

### Megjegyzés
Ez a modell szolgál a projekt **első komoly referencia megoldásaként**,  
amelyhez a későbbi architektúrák (pl. DenseNet, EfficientNet, PyTorch-alapú modellek)  
eredményei összehasonlíthatók.