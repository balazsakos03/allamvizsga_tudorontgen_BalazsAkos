# 📊 Kísérleti eredmények és modellek

Ez a mappa a projekt során megvalósított neurális hálózatok  
**eredményeit, kiértékeléseit és tapasztalatait** tartalmazza.

A cél nem egyetlen modell bemutatása, hanem a **modellek fejlődési folyamatának**,  
valamint azok **objektív összehasonlításának** dokumentálása.

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
Az itt elért eredmények elsősorban **kiindulási alapként** szolgálnak.

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

### Kvantitatív eredmények (teszt halmaz)
- **Accuracy:** 86.54%
- **ROC–AUC:** 0.943
- **PNEUMONIA recall:** 94.87%
- **NORMAL recall:** 72.65%

A confusion matrix alapján megfigyelhető, hogy a modell  
**különösen jól teljesít a pneumonia esetek felismerésében**,  
ugyanakkor a NORMAL osztály esetén több téves pozitív predikció fordul elő.

### Megjegyzés
Ez a modell szolgál a projekt **első komoly referencia megoldásaként**,  
amelyhez a további architektúrák teljesítménye összehasonlítható.

---

## 3️⃣ Transfer Learning alapú modell – DenseNet121

**Notebook:**  
`notebooks/03_transfer_learning_densenet.ipynb`

### Modell leírása
A harmadik kísérlet a **DenseNet121 architektúrára** épül,  
amely sűrű összeköttetéseinek köszönhetően hatékonyabb feature-újrahasznosítást tesz lehetővé.

A tanítási stratégia megegyezik a ResNet50 modellnél alkalmazott módszertannal:
- ImageNet előtanított súlyok,
- adataugmentáció,
- osztályegyensúly kezelése,
- kétfázisú tanítás (fagyasztás + fine-tuning).

### Kvantitatív eredmények (teszt halmaz)
- **Accuracy:** 89.74%
- **ROC–AUC:** 0.965
- **PNEUMONIA recall:** 93.08%
- **NORMAL recall:** 84.19%

A confusion matrix alapján a DenseNet121:
- kevesebb téves pozitív predikciót eredményez a NORMAL osztályban,
- kiegyensúlyozottabb teljesítményt mutat mindkét osztály esetén,
- összességében magasabb általános pontosságot ér el.

---

## 📈 Modellek összehasonlítása

| Modell       | Accuracy | ROC–AUC | NORMAL Recall | PNEUMONIA Recall |
|-------------|----------|---------|---------------|------------------|
| ResNet50    | 86.54%   | 0.943   | 72.65%        | 94.87%           |
| DenseNet121 | 89.74%   | 0.965   | 84.19%        | 93.08%           |

### Értelmezés
- A **DenseNet121** modell összességében **jobb általános teljesítményt** mutat.
- A **ResNet50** erőssége a pneumonia esetek nagyon magas recall értéke.
- A DenseNet kiegyensúlyozottabb döntéshozatala különösen előnyös lehet  
  olyan környezetben, ahol a téves riasztások csökkentése is fontos szempont.

---

## 🔍 Kvalitatív kiértékelés
A kvantitatív metrikák mellett mindkét modell esetében  
**vizuális elemzés is készült** a teszt halmaz képein.

A vizualizációk:
- helyes és hibás predikciókat egyaránt bemutatnak,
- lehetőséget adnak a modellek döntéseinek értelmezésére,
- segítenek azonosítani a tipikus tévesztési eseteket.

A kapcsolódó ábrák a `results/figures/` mappában találhatók.

### Grad-CAM (magyarázhatóság)
A modellek döntéseinek értelmezhetősége érdekében **Grad-CAM** (Gradient-weighted Class Activation Mapping)
vizualizációkat is készítettem, amelyek megmutatják, hogy az osztályozás során a hálózat mely képrégiókra fókuszált.

A Grad-CAM overlay képek a következő mappákban találhatók:
- `results/figures/02_transfer_learning_resnet/gradcam_resnet50/`
- `results/figures/03_transfer_learning_densenet/gradcam_densenet121/`