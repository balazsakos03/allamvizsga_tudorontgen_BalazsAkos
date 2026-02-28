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

## 4️⃣ Transfer Learning alapú modell – EfficientNetB0

**Notebook:**  
`notebooks/04_transfer_learning_efficientnetb0.ipynb`

### Modell leírása
A negyedik kísérlet az **EfficientNetB0 architektúrát** alkalmazza,  
amely a számítási hatékonyságot és a teljesítményt együttesen optimalizáló  
**compound scaling** megközelítésre épül.

A modell:
- ImageNet-en előtanított súlyokat használ,
- EfficientNet-specifikus előfeldolgozást alkalmaz,
- kétfázisú tanítással került optimalizálásra.

### Kvantitatív eredmények (teszt halmaz)
- **Accuracy:** 87.34%
- **ROC–AUC:** 0.952
- **PNEUMONIA recall:** 94.10%
- **NORMAL recall:** 76.07%

Az eredmények alapján az EfficientNetB0:
- stabil teljesítményt nyújt mindkét osztály esetén,
- jobb általános pontosságot ér el, mint a ResNet50,
- ugyanakkor kevésbé kiegyensúlyozott, mint a DenseNet121.

---

## 5️⃣ Transfer Learning alapú modell – MobileNetV2

**Notebook:**  
`notebooks/05_transfer_learning_mobilenetv2.ipynb`

### Modell leírása
Az ötödik kísérlet a **MobileNetV2 architektúrát** alkalmazza,  
amely kifejezetten alacsony számítási igényű, mobil és edge környezetekre tervezett modell.

A MobileNetV2:
- depthwise separable convolúciókat használ,
- jelentősen kevesebb paraméterrel rendelkezik, mint a klasszikus CNN-ek,
- mégis képes versenyképes teljesítményt nyújtani komplex vizuális feladatokon.

A tanítás során:
- ImageNet-en előtanított súlyokat használtam,
- MobileNet-specifikus előfeldolgozást alkalmaztam,
- kétfázisú tanítást végeztem (fagyasztás + finomhangolás).

### Kvantitatív eredmények (teszt halmaz)
- **Accuracy:** 90.38%
- **ROC–AUC:** 0.964
- **PNEUMONIA recall:** 91.03%
- **NORMAL recall:** 89.32%

A confusion matrix alapján megfigyelhető, hogy a MobileNetV2:
- mindkét osztály esetén magas recall értéket ér el,
- különösen kiegyensúlyozott teljesítményt mutat,
- a legmagasabb összpontosságot biztosítja a vizsgált modellek közül,
  annak ellenére, hogy számítási igénye jelentősen alacsonyabb.

---

## 6️⃣ Transfer Learning alapú modell – VGG16

**Notebook:**
`notebooks/06_transfer_learning_vgg16.ipynb`

### Modell leírása
A hatodik kísérlet a **VGG16 architektúrát** alkalmazza, amely egy klasszikus, mély konvolúciós neurális hálózat.

A VGG16:
- egymásra épülő 3×3 konvolúciós rétegeket használ,
- nem tartalmaz residual vagy dense összeköttetéseket,
- jelentősen nagyobb paraméterszámmal rendelkezik, mint a modern, optimalizált architektúrák.

A tanítás során:
- ImageNet-en előtanított súlyokat használtam,
- VGG-specifikus előfeldolgozást alkalmaztam,
- kétfázisú tanítást végeztem (fagyasztás + fine-tuning),
- osztályegyensúly kezelést alkalmaztam.

### Kvantitatív eredmények (teszt halmaz, threshold = 0.5)

- **Accuracy:** 88.62%
- **ROC–AUC:** 0.9726
- **PNEUMONIA recall:** 98.72%
- **NORMAL recall:** 71.79%

A confusion matrix alapján megfigyelhető, hogy a modell:
- rendkívül magas pneumonia felismerési arányt ér el,
- ugyanakkor jelentős számú egészséges esetet sorol tévesen beteg kategóriába.

### Megjegyzés

A VGG16 kiemelkedően magas ROC–AUC értéket produkált, ami azt jelzi, hogy a modell nagyon jól szeparálja a két osztályt, azonban az alapértelmezett 0.5-ös döntési küszöb mellett a NORMAL osztály felismerése kevésbé kiegyensúlyozott.

---

## 7️⃣ Transfer Learning alapú modell – VGG16 (módosított threshold)

**Notebook:**
`notebooks/07_transfer_learning_modified_vgg16.ipynb`

### Modell leírása

Ebben a kísérletben a VGG16 architektúrát változatlan formában alkalmaztam, azonban a döntési küszöb (classification threshold) optimalizálásra került.

A cél:
- a sensitivity és specificity közötti egyensúly javítása,
- a NORMAL osztály felismerési arányának növelése,
- a klinikailag releváns kompromisszum vizsgálata.

A tanítás teljes folyamatát újrafuttattam, majd a döntési küszöb módosításával végeztem kiértékelést.

### Kvantitatív eredmények (teszt halmaz, módosított threshold)

- **Accuracy:** 90.54%
- **ROC–AUC:** 0.9748
- **PNEUMONIA recall:** 97.69%
- **NORMAL recall:** 78.63%

A confusion matrix alapján megfigyelhető, hogy:
- a NORMAL osztály recall értéke jelentősen javult,
- a pneumonia felismerési arány minimálisan csökkent,
- az overall accuracy növekedett.

### Megjegyzés

A threshold optimalizáció eredményeként a modell:
- kiegyensúlyozottabb teljesítményt mutat,
- magas szeparációs képességét (ROC–AUC) megőrizte,
- klinikai szempontból rugalmasabban paraméterezhetővé vált.

Ez a kísérlet rávilágít arra, hogy a neurális hálózat teljesítménye nem kizárólag az architektúrától, hanem a döntési stratégia megválasztásától is függ.

---

## 📈 Modellek összehasonlítása

| Modell         | Accuracy | ROC–AUC | NORMAL Recall | PNEUMONIA Recall |
|----------------|----------|---------|---------------|------------------|
| ResNet50       | 86.54%   | 0.943   | 72.65%        | 94.87%           |
| DenseNet121    | 89.74%   | 0.965   | 84.19%        | 93.08%           |
| EfficientNetB0 | 87.34%   | 0.952   | 76.07%        | 94.10%           |
|**MobileNetV2** | 90.38% | 0.964 |**89.32%**     |  91.03%       |
| VGG16(0.5 threshold) | 88.62% | 0.973 | 71.79% | **98.2%** |
| VGG16(optimal threshold) | **90.54%** | **0.975** | 78.63% | 97.69% |

### Értelmezés
- A **MobileNetV2** a legmagasabb **overall accuracy** értéket érte el, miközben alacsony számítási igényű architektúra.
- A **DenseNet121** továbbra is az egyik legkiegyensúlyozottabb modell, különösen a ROC–AUC érték tekintetében.
- A **ResNet50** és **EfficientNetB0** modellek erőssége a pneumonia esetek magas felismerési aránya, azonban a NORMAL osztályban több tévesztést mutatnak.
- A MobileNetV2 eredményei azt jelzik, hogy **könnyű, erőforrás-hatékony modellek is képesek klinikailag releváns teljesítményt nyújtani**.
- A **VGG16** érte el a legmagasabb ROC–AUC értéket, ami a legjobb osztályszeparációra utal. Az alapértelmezett 0.5-ös threshold mellett a modell erősen pneumonia-orientált döntést hozott. A döntési küszöb optimalizálása jelentősen javította a NORMAL osztály felismerését, miközben a pneumonia detektálási arány klinikailag releváns tartományban maradt. A **MobileNetV2** továbbra is a legkiegyensúlyozottabb és paraméter-hatékony megoldás. A kísérletek eredményei azt mutatják, hogy a modell viselkedését nemcsak az architektúra, hanem a döntési stratégia is jelentősen befolyásolja.

---

## 🔍 Kvalitatív kiértékelés és magyarázhatóság

A kvantitatív metrikák mellett mindhárom Transfer Learning modell esetében  
**vizuális és kvalitatív elemzés is készült** a teszt halmaz képein.

### Grad-CAM (magyarázhatóság)
A **Grad-CAM (Gradient-weighted Class Activation Mapping)** módszer segítségével  
vizualizáltam, hogy a modellek döntéshozatala során mely képrégiók járultak hozzá  
leginkább az osztályozási eredményekhez.

A Grad-CAM overlay képek az alábbi mappákban találhatók:
- `results/figures/02_transfer_learning_resnet/gradcam_resnet50/`
- `results/figures/03_transfer_learning_densenet/gradcam_densenet121/`
- `results/figures/04_transfer_learning_efficientnetb0/gradcam_efficientnetb0/`
- `results/figures/05_transfer_learning_mobilenetv2/gradcam_mobilenetv2/`

A vizualizációk segítik a modellek döntéseinek értelmezését,  
valamint rávilágítanak az esetleges tévesztések mögötti mintázatokra.