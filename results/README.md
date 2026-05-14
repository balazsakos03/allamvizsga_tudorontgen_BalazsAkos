# Kísérleti eredmények és modellek

Ez a mappa a projekt során megvalósított neurális hálózatok eredményeit,
kiértékeléseit és tapasztalatait tartalmazza.

A cél nem egyetlen modell bemutatása, hanem a modellek fejlődési folyamatának
és objektív összahasonlításának dokumentálása — beleértve a sikertelen
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

## 11. Transfer Learning – VGG16 + crop + CLAHE (COVID-19 Radiography Database)

**Notebook:** `notebooks/11-transfer-learning-vgg16-crop-clahe.ipynb`

Ez a kísérlet a 10-es notebookkal azonos architektúrát és preprocessing
stratégiát alkalmaz, azonban egy teljesen más forrásból származó adatkészleten:
a COVID-19 Radiography Database-en (Kaggle, Chowdhury et al.).

A dataset a 10-es kísérletben használt Kaggle Chest X-Ray datasettől több
szempontból is eltér. Képei felnőtt betegekről készültek, homogénebb
minőségűek, és jelentősen nagyobb Normal osztályt tartalmaznak (10,192 Normal
vs. 1,345 Viral Pneumonia). Mivel a dataset nem tartalmaz előre felosztott
train/val/test struktúrát, a képeket 80/10/10 arányban osztottuk fel
stratifikált mintavétellel. A COVID és Lung Opacity osztályokat kihagytuk,
kizárólag a Normal és Viral Pneumonia képeket használtuk.

A kísérlet elsődleges célja a cross-dataset generalizáció vizsgálatának
előkészítése: egy második, független forrásból tanított modell rendelkezésre
állása lehetővé teszi annak mérését, hogy a két modell mennyire teljesít
egymás teszthalmazán.

### Eredmények

- **Accuracy:** 98.61%
- **ROC–AUC:** 0.9987
- **PNEUMONIA recall:** 92.54%
- **NORMAL recall:** 99.41%

A kiemelkedően magas metrikák részben a dataset homogénebb képminőségének
és a nagyobb Normal osztálynak köszönhetők. A ROC–AUC értéke 0.9987,
amely gyakorlatilag tökéletes osztályszeparációra utal. A cross-dataset
kiértékelés fogja megmutatni, hogy ez a teljesítmény mennyire generalizál
idegen forrásból származó felvételekre.

---

## 12. Cross-Dataset kiértékelés

**Notebook:** `notebooks/12-cross-dataset-evaluation.ipynb`

A 10-es és 11-es kísérletben betanított két modell keresztbe tesztelése
egymás teszthalmazán. A cél annak mérése, hogy a modellek mennyire
generalizálnak idegen forrásból származó felvételekre.

A négy kiértékelt kombináció:

| Modell | Tesztelve | Accuracy | ROC–AUC | NORMAL Recall | PNEUMONIA Recall |
|--------|-----------|----------|---------|---------------|------------------|
| VGG16 (Kaggle) | Kaggle (saját) | 91.67% | 0.9742 | 83.33% | 96.67% |
| VGG16 (Kaggle) | COVID Radiography (idegen) | 56.76% | 0.8360 | 51.76% | 94.78% |
| VGG16 (COVID Radiography) | COVID Radiography (saját) | 98.61% | 0.9987 | 99.41% | 92.54% |
| VGG16 (COVID Radiography) | Kaggle (idegen) | 84.29% | 0.9685 | 58.97% | 99.49% |

### Megfigyelések

Mindkét modell jelentős teljesítménycsökkenést mutat idegen dataseten,
ami a domain shift probléma jelenlétét igazolja. A két modell azonban
eltérő mértékben generalizál:

A Kaggle-n tanított modell idegen dataseten 91.67%-ról 56.76%-ra esik
vissza — ez drasztikus csökkenés. A modell szinte mindent pneumoniának
minősít (PNEUMONIA recall: 94.78%), miközben a normál esetek felét
tévesen betegnek ítéli (NORMAL recall: 51.76%).

A COVID Radiography-n tanított modell robusztusabbnak bizonyul: idegen
dataseten 98.61%-ról 84.29%-ra esik vissza, és a ROC–AUC értéke
idegen halmazon is magas marad (0.9685). A gyengesége itt is a NORMAL
osztály alacsony recall értéke (58.97%), azonban ez a viselkedés
részben a tanítóhalmaz erős osztályegyensúlytalanságából ered
(10,192 Normal vs. 1,345 Pneumonia).

Fontos megfigyelés, hogy mindkét modell NORMAL precision értéke idegen
dataseten is magas marad (Kaggle modell: 98.69%, COVID modell: 98.57%),
ami azt jelenti: amit normálnak ítél, azt nagy biztonsággal helyesen
ítéli — a probléma a téves pozitív pneumonia predikciók magas száma.

A cross-dataset ROC–AUC értékek (0.836 és 0.969) azt mutatják, hogy
az osztályszeparációs képesség megmarad idegen képeken is — a döntési
küszöb optimalizálásával mindkét modell teljesítménye javítható lenne
az idegen dataseten is.

---

## 13. Transfer Learning – VGG16 + crop + CLAHE (Kombinált adatkészlet)

**Notebook:** `notebooks/13-transfer-learning-vgg16-combined-dataset.ipynb`

A cross-dataset kiértékelés tanulságai alapján a következő lépés két
különböző forrásból származó adatkészlet összevonása volt. A kombinált
dataset a Kaggle Chest X-Ray és a COVID-19 Radiography Database képeiből
áll össze, lokálisan a `datasets/merge_datasets.py` szkript segítségével
kerültek összeállításra.

Az összevonás során undersampling kerül alkalmazásra: a Normal osztály
a Pneumonia osztály méretére lett csökkentve, így a tanítóhalmaz teljesen
kiegyensúlyozott (4,494 Normal + 4,494 Pneumonia a train halmazban).
A dataset összesen 11,236 képet tartalmaz, train/val/test arányban 80/10/10
felosztással.

Az architektúra és a preprocessing azonos a 10-es kísérlettel
(VGG16 + aszimmetrikus crop + CLAHE), kétfázisú tanítással.

### Eredmények

- **Accuracy:** 99.11%
- **ROC–AUC:** 0.9984
- **PNEUMONIA recall:** 99.47%
- **NORMAL recall:** 98.75%

Ez az első kísérlet ahol mindkét osztály recall értéke 98% felett van
egyidejűleg. A confusion matrix alapján 562 Normal képből mindössze 7,
562 Pneumonia képből mindössze 3 került tévesen besorolásra. A kombinált
dataset tehát egyszerre kezeli a class imbalance és a domain shift
problémát, mivel a tanítóhalmaz különböző forrásokból, különböző
készülékekkel és különböző betegcsoportokon készült felvételeket egyaránt
tartalmaz.

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
| VGG16 + crop + CLAHE (Kaggle) | 91.67% | 0.9742 | 83.33% | 96.67% |
| VGG16 + crop + CLAHE (COVID Radiography) | 98.61% | 0.9987 | 99.41% | 92.54% |
| VGG16 + crop + CLAHE (Combined) | **99.11%** | **0.9984** | **98.75%** | **99.47%** |

### Értelmezés

A modellek fejlődési íve jól látható a táblázatból. A Transfer Learning alapú
architektúrák mindegyike érdemben felülmúlja a baseline CNN-t. A MobileNetV2
kiemelkedik kiegyensúlyozottságával és alacsony számítási igényével. A VGG16
variánsok a legmagasabb ROC–AUC értékeket mutatják, ami erős osztályszeparációra
utal. A CLAHE és crop alapú előfeldolgozás nem csak a teszthalmazon mért
metrikákat javította, hanem érdemben növelte a modellek robusztusságát
idegen, valós körülmények között készült felvételeken is.

A COVID-19 Radiography Database-en tanított modell kiemelkedő metrikái
részben a dataset kedvezőbb tulajdonságainak (homogén képminőség, nagy
Normal osztály) köszönhetők, ezért az eredmények nem vethetők össze
közvetlenül a Kaggle dataseten mért értékekkel. A két modell egymás
teszthalmazán mért teljesítménye a cross-dataset kiértékelés keretében
kerül bemutatásra.

Az U-Net alapú kísérlet rávilágított arra, hogy egy összetettebb preprocessing
pipeline önmagában nem garantál jobb eredményt, ha maga a pipeline is
adatfüggő és domain shift-re érzékeny komponenst tartalmaz.

A cross-dataset kiértékelés megerősítette, hogy a saját teszthalmazon
mért teljesítmény nem jelzi előre megbízhatóan az idegen képeken való
viselkedést. A COVID Radiography-n tanított modell bizonyult
robusztusabbnak, azonban mindkét modell esetén a domain shift hatása
egyértelműen kimutatható.

A kombinált adatkészleten tanított modell érte el a projekt legjobb
eredményeit: 99.11% accuracy és 0.9984 ROC–AUC értékkel, miközben
mindkét osztályban 98% feletti recall értéket produkált. Ez megerősíti,
hogy a domain shift probléma leghatékonyabb kezelése nem a preprocessing
bonyolításában, hanem a tanítóhalmaz diverzitásának növelésében rejlik.

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
- `results/figures/11_transfer_learning_vgg16_crop_clahe/gradcam_vgg16_covid/`
- `results/figures/13_transfer_learning_vgg16_combined/gradcam_vgg16_combined/`