# Kísérleti eredmények és modellek

Ez a mappa a projekt során megvalósított neurális hálózatok eredményeit,
kiértékeléseit és tapasztalatait tartalmazza.

A cél nem egyetlen modell bemutatása, hanem a modellek fejlődési folyamatának
és objektív összehasonlításának dokumentálása — beleértve a sikertelen
megközelítéseket is, amelyek tanulságai szintén részét képezik a dolgozatnak.

---

## 1. Baseline CNN

**Notebook:** `notebooks/01-baseline-cnn.ipynb`

Az első modell egy egyszerű, előtanított súlyokat nem alkalmazó konvolúciós
neurális háló, amelynek célja a feldolgozási pipeline kipróbálása és egy
kiindulási referenciapont létrehozása a későbbi modellekhez.

A modell nem tartalmaz finomhangolást vagy komplex regularizációt,
eredményei elsősorban összehasonlítási alapként szolgálnak.

---

## 2. Transfer Learning – ResNet50

**Notebook:** `notebooks/02-transfer-learning-resnet.ipynb`

ImageNet-en előtanított ResNet50 architektúra, kétfázisú tanítással
(fagyasztott backbone, majd fine-tuning), adataugmentációval és
osztályegyensúly kezeléssel.

### Eredmények

- **Accuracy:** 86.54%
- **ROC–AUC:** 0.943
- **PNEUMONIA recall:** 94.87%
- **NORMAL recall:** 72.65%

A modell jól teljesít a pneumonia esetek felismerésében, azonban
a NORMAL osztályban több téves pozitív predikció fordul elő.
Ez a projekt első komolyabb referencia megoldása.

---

## 3. Transfer Learning – DenseNet121

**Notebook:** `notebooks/03-transfer-learning-densenet.ipynb`

DenseNet121 architektúra, amely sűrű összeköttetéseinek köszönhetően
hatékonyabb feature-újrahasznosítást tesz lehetővé. A tanítási stratégia
megegyezik a ResNet50 modellnél alkalmazottal.

### Eredmények

- **Accuracy:** 89.74%
- **ROC–AUC:** 0.965
- **PNEUMONIA recall:** 93.08%
- **NORMAL recall:** 84.19%

A ResNet50-hez képest kiegyensúlyozottabb teljesítmény, kevesebb téves
pozitív predikcióval a NORMAL osztályban.

---

## 4. Transfer Learning – EfficientNetB0

**Notebook:** `notebooks/04-transfer-learning-efficientnetb0.ipynb`

EfficientNetB0 architektúra, amely compound scaling megközelítéssel
együttesen optimalizálja a mélységet, szélességet és felbontást.

### Eredmények

- **Accuracy:** 87.34%
- **ROC–AUC:** 0.952
- **PNEUMONIA recall:** 94.10%
- **NORMAL recall:** 76.07%

Stabil teljesítmény, a ResNet50-nél jobb accuracy, azonban kevésbé
kiegyensúlyozott mint a DenseNet121.

---

## 5. Transfer Learning – MobileNetV2

**Notebook:** `notebooks/05-transfer-learning-mobilenetv2.ipynb`

MobileNetV2 architektúra, amely depthwise separable konvolúciókat
alkalmazva kifejezetten alacsony számítási igényű, mobil és edge
környezetekre tervezett modell.

### Eredmények

- **Accuracy:** 90.38%
- **ROC–AUC:** 0.964
- **PNEUMONIA recall:** 91.03%
- **NORMAL recall:** 89.32%

A vizsgált modellek közül a legkiegyensúlyozottabb teljesítmény,
mindkét osztályban magas recall értékkel — annak ellenére, hogy
paramétereinek száma és számítási igénye jelentősen alacsonyabb
a többi architektúránál.

---

## 6. Transfer Learning – VGG16

**Notebook:** `notebooks/06-transfer-learning-vgg16.ipynb`

Klasszikus VGG16 architektúra, egymásra épülő 3×3 konvolúciós rétegekkel.
Jelentősen nagyobb paraméterszámmal rendelkezik mint a modern,
optimalizált architektúrák.

### Eredmények (threshold = 0.5)

- **Accuracy:** 88.62%
- **ROC–AUC:** 0.9726
- **PNEUMONIA recall:** 98.72%
- **NORMAL recall:** 71.79%

Kiemelkedően magas ROC–AUC érték, amely erős osztályszeparációra utal.
Az alapértelmezett 0.5-ös küszöb mellett azonban a modell erősen
pneumonia-orientált döntést hoz, és a NORMAL osztályban jelentős
számú téves pozitív predikció fordul elő.

---

## 7. Transfer Learning – VGG16 (optimalizált threshold)

**Notebook:** `notebooks/07-transfer-learning-modified-vgg16.ipynb`

A VGG16 architektúra változatlan formában, azonban a döntési küszöb
szisztematikus optimalizálásával. A cél a sensitivity és specificity
közötti egyensúly javítása, klinikailag releváns kompromisszum elérése.

### Eredmények (optimalizált threshold)

- **Accuracy:** 90.54%
- **ROC–AUC:** 0.9748
- **PNEUMONIA recall:** 97.69%
- **NORMAL recall:** 78.63%

A NORMAL osztály recall értéke jelentősen javult, a pneumonia felismerési
arány klinikailag releváns tartományban maradt. Ez a kísérlet rávilágít arra,
hogy a modell viselkedését nemcsak az architektúra, hanem a döntési
stratégia megválasztása is jelentősen befolyásolja.

---

## 8. Transfer Learning – VGG16 + CLAHE

**Notebook:** `notebooks/08-transfer-learning-vgg16-clahe.ipynb`

Az előző kísérletek eredményei alapján a VGG16 architektúrát választottam
a további fejlesztések alapjául, mivel metrikái alapján stabilabban teljesít
a többi architektúránál. Ebben a kísérletben CLAHE (Contrast Limited Adaptive
Histogram Equalization) előfeldolgozást alkalmaztam, amely a röntgenfelvételek
eltérő kontrasztját és fényerejét kiegyenlíti.

A motiváció: az eddigi modellek webappon, valós idejű tesztelés során idegen
képeken gyenge eredményeket mutattak. A CLAHE célja, hogy a bemeneti kép
kontrasztját a háló által már ismert tartományba hozza, ezzel javítva az
idegen képeken való teljesítményt.

### Eredmények

- **Accuracy:** 91.83%
- **ROC–AUC:** 0.9756
- **PNEUMONIA recall:** 97.44%
- **NORMAL recall:** 82.48%

A CLAHE előfeldolgozás mérhető javulást hozott az idegen képeken való
teljesítményben, azonban a domain shift problémát nem oldotta meg teljesen.

---

## 9. Transfer Learning – VGG16 + U-Net szegmentáció + CLAHE

**Notebook:** `notebooks/09-transfer-learning-vgg16-unet-clahe.ipynb`

A domain shift probléma kezelésére ebben a kísérletben egy U-Net alapú
tüdőszegmentációt alkalmaztam preprocessing lépésként. Az elképzelés az volt,
hogy ha a háló csak a tüdő területét látja, a háttér és a képen lévő egyéb
elemek (feliratok, kábelek, műtermékek) nem zavarják a döntéshozatalt.

A szegmentációs modell egy nyilvánosan elérhető, előtanított U-Net volt
(`unet_lung_seg.hdf5`). A pipeline: U-Net maszk → masked crop → CLAHE → VGG16.

### A megközelítés korlátai

Az U-Net modell azonban más adatkészleten tanult, mint a saját felvételek,
ezért rajta is megjelent a domain shift probléma: az általa generált maszkok
több képen pontatlanok vagy tévesek voltak, különösen a dataset gyermekbetegeknél
készült felvételein, ahol a tüdő formája és mérete eltér a felnőtt referenciaképektől.

### Eredmények

- **Accuracy:** 91.51%
- **ROC–AUC:** 0.9682
- **PNEUMONIA recall:** 94.62%
- **NORMAL recall:** 86.32%

A metrikák nem javultak érdemben a CLAHE-only verzióhoz képest, a rossz
maszkok miatt a modell helyenként irreleváns képrészleteken tanult.
Ez a kísérlet fontos tanulságként szerepel a dolgozatban: egy szofisztikált
preprocessing pipeline nem feltétlenül jobb, ha maga a pipeline is domain
shift-re érzékeny.

---

## 10. Transfer Learning – VGG16 + aszimmetrikus crop + CLAHE

**Notebook:** `notebooks/10-transfer-learning-vgg16-crop-clahe.ipynb`

Az U-Net alapú megközelítés tapasztalatai alapján egy egyszerűbb, de
robusztusabb preprocessing stratégiát alkalmaztam: aszimmetrikus center crop,
amelyet CLAHE követ. A crop arányok a tüdőröntgenek tipikus elrendezéséhez
igazodnak (felülről több margó kerül levágásra a váll és nyak területe miatt).

A megközelítés előnye, hogy determinisztikus, nem tartalmaz ML komponenst,
így nem tud domain shift-re érzékennyé válni. A crop paraméterek vizuális
sanity check alapján kerültek beállításra.

Crop paraméterek: top=15%, bottom=5%, left=5%, right=5%.

### Eredmények

- **Accuracy:** 91.67%
- **ROC–AUC:** 0.9742
- **PNEUMONIA recall:** 96.67%
- **NORMAL recall:** 83.33%

A metrikák a CLAHE-only verzióval összevethetők, azonban ez a megoldás
lényegesen megbízhatóbb idegen képeken, mivel a preprocessing nem tartalmaz
adatfüggő komponenst.

---

## Modellek összehasonlítása

| Modell | Accuracy | ROC–AUC | NORMAL Recall | PNEUMONIA Recall |
|--------|----------|---------|---------------|------------------|
| Baseline CNN | – | – | – | – |
| ResNet50 | 86.54% | 0.943 | 72.65% | 94.87% |
| DenseNet121 | 89.74% | 0.965 | 84.19% | 93.08% |
| EfficientNetB0 | 87.34% | 0.952 | 76.07% | 94.10% |
| MobileNetV2 | 90.38% | 0.964 | 89.32% | 91.03% |
| VGG16 (0.5 threshold) | 88.62% | 0.973 | 71.79% | 98.72% |
| VGG16 (opt. threshold) | 90.54% | 0.975 | 78.63% | 97.69% |
| VGG16 + CLAHE | 91.83% | 0.9756 | 82.48% | 97.44% |
| VGG16 + U-Net + CLAHE | 91.51% | 0.9682 | 86.32% | 94.62% |
| VGG16 + crop + CLAHE | 91.67% | 0.9742 | 83.33% | 96.67% |

### Értelmezés

A modellek fejlődési íve jól látható a táblázatból. A Transfer Learning alapú
architektúrák mindegyike érdemben felülmúlja a baseline CNN-t. A MobileNetV2
kiemelkedik kiegyensúlyozottságával és alacsony számítási igényével. A VGG16
variánsok a legmagasabb ROC–AUC értékeket mutatják, ami erős osztályszeparációra
utal. A CLAHE és crop alapú előfeldolgozás nem csak a teszthalmazon mért
metrikákat javította, hanem érdemben növelte a modellek robusztusságát
idegen, valós körülmények között készült felvételeken is.

Az U-Net alapú kísérlet rávilágított arra, hogy egy összetettebb preprocessing
pipeline önmagában nem garantál jobb eredményt, ha maga a pipeline is
adatfüggő és domain shift-re érzékeny komponenst tartalmaz.

---

## Kvalitatív kiértékelés és magyarázhatóság

A kvantitatív metrikák mellett minden Transfer Learning modell esetében
vizuális kiértékelés is készült Grad-CAM (Gradient-weighted Class Activation
Mapping) segítségével, amely megmutatja, hogy a modell döntéshozatala során
mely képrégiók járultak hozzá leginkább az osztályozási eredményhez.

A Grad-CAM overlay képek az alábbi mappákban találhatók:

- `results/figures/02_transfer_learning_resnet/gradcam_resnet50/`
- `results/figures/03_transfer_learning_densenet/gradcam_densenet121/`
- `results/figures/04_transfer_learning_efficientnetb0/gradcam_efficientnetb0/`
- `results/figures/05_transfer_learning_mobilenetv2/gradcam_mobilenetv2/`
- `results/figures/06_transfer_learning_vgg16/gradcam_vgg16/`
- `results/figures/07_transfer_learning_vgg16_threshold/gradcam_vgg16_threshold/`
- `results/figures/08_transfer_learning_vgg16_clahe/gradcam_vgg16_clahe/`
- `results/figures/09_transfer_learning_vgg16_unet_clahe/gradcam_vgg16_unet_clahe/`
- `results/figures/10_transfer_learning_vgg16_crop_clahe/gradcam_vgg16_crop_clahe/`