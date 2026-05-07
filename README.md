# Tüdőröntgen felvételek osztályozása mély tanulású neurális hálózatokkal

Szakdolgozat – Balázs Ákos – 2025/26

---

## A projektről

A dolgozat célja mellkasröntgen felvételek automatikus osztályozása mélytanulási módszerekkel, valamint különböző neurális hálózati architektúrák és előfeldolgozási stratégiák szisztematikus összehasonlítása.

A hangsúly nem egyetlen végső modell megalkotásán van, hanem a fejlesztési folyamat dokumentálásán: minden kísérlet tanulsága — beleértve a sikertelen megközelítéseket is — részét képezi a dolgozatnak.

---

## Megközelítés

A projekt iteratív felépítésű. Kiindulópontként egy egyszerű baseline CNN kerül megvalósításra, amelyre fokozatosan összetettebb Transfer Learning alapú architektúrák épülnek. Az egyes kísérletek eredményei egységes metrikakészlettel kerülnek összehasonlításra (Accuracy, ROC-AUC, Sensitivity, Specificity).

A vizsgált architektúrák: baseline CNN, ResNet50, DenseNet121, EfficientNetB0, MobileNetV2, VGG16 és annak több előfeldolgozási variánsa (CLAHE, U-Net szegmentáció, aszimmetrikus crop).

Az eredmények és tapasztalatok részletes leírása a `results/README.md` fájlban található.

---

## Repository felépítése
```
├── data/               # Felhasznált adatkészletek dokumentációja
├── notebooks/          # Tanítási és kiértékelési notebookok
├── results/            # Modellek metrikái és kiértékelési grafikonok
├── test_pictures/      # Idegen képek amelyekkel a hálókat lehet tesztelni a webapp keretében
├── webapp/             # Streamlit webapp a vizualizáció és valós idejű tesztelés céljából
├── .gitignore
└── README.md
```

Az adatkészletek nem kerülnek feltöltésre, csak dokumentálásra — lásd: `data/README.md`.

---

## Technológiák

- Python 3.10+
- TensorFlow / Keras
- NumPy, OpenCV, Matplotlib, scikit-learn
- Kaggle Notebook (fő futtatási platform)

---

## Szerző

Szakdolgozat – Balázs Ákos – 2025/26