# Tüdőröntgen Elemző Webalkalmazás 🫁

Ez a mappa tartalmazza a diplomamunkámhoz készült Streamlit webalkalmazást. A felület célja a különböző mélytanulású (Deep Learning) konvolúciós neurális hálózatok (ResNet50, DenseNet121, EfficientNetB0, MobileNetV2, VGG16) teljesítményének összehasonlítása, valamint valós idejű kiértékelés biztosítása új tüdőröntgen felvételeken.

## 🌟 Fő funkciók

* 📊 **Modell Statisztikák:** A betanított hálózatok metrikáinak (Accuracy, ROC AUC, F1-score) automatikus beolvasása és vizuális összehasonlítása interaktív grafikonokon.
* ⚡ **Valós idejű predikció (Inference):** Felhasználó által feltöltött röntgenképek azonnali elemzése a kiválasztott modellt alkalmazva (Normál vs. Tüdőgyulladás).
* 🔍 **Grad-CAM Vizualizáció:** Az "fekete doboz" jelenség csökkentése érdekében a rendszer hőtérképet (heatmap) generál, amely megmutatja, a hálózat a tüdő mely részeire fókuszált a predikció során.
* ⚙️ **Intelligens Előfeldolgozás:** A kód a kiválasztott modell alapján automatikusan alkalmazza a szükséges előfeldolgozást (pl. standardizáció, modell-specifikus `preprocess_input`, vagy LAB színteres **CLAHE** kontrasztjavítás).

## 📂 Mappaszerkezet

```text
webapp/
├── app.py                 # A Streamlit alkalmazás fő kódja
├── metrics/               # A hálózatok kiértékelési metrikái (.json formátumban)
└── models/                # A betanított súlyok helye (.keras formátumban)
```

**Megjegyzés:** A GitHub fájlméret-korlátozásai miatt a betanított modellek súlyai (.keras fájlok) nincsenek feltöltve a tárolóba. Ezeket lokálisan kell a models/ mappába helyezni a futtatás előtt.

## Telepítés és Futtatás lokálisan
# 1. Nyisd meg a terminált a webapp mappában.
# 2. Készíts és aktiválj egy virtuális környezetet:
```python
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```
# 3.Telepítsd a szükséges könyvtárakat:
```bash
pip install streamlit tensorflow pandas pillow opencv-python matplotlib
```
# 4.Indítsd el az alkalmazást:
```bash
streamlit run app.py
```
A böngésző automatikusan megnyílik a http://localhost:8501 címen.