# ErgoDesk_clasificador_de_posturas

## Autores: Mario Vargas y Yeison García. 
## Afiliación: Universidad de La Salle

Proyecto en el área de computer visión, usando Medipipe y algoritmos de clasificación (RF, SVM y LR)
Los scripts acá detallados, soportan el EDA y preprocesmaiento de imagenes de posturas en escritorios de trabajo. Así mismo: 
- La carpeta modelo_mediapipe: contiene los scripts para extraer features con media pipe de las imagenes de entrenamiento, entrenar 3 modelos de clasificación como son Random Forest, regresión logística y un script de validación.
- La carpeta producción: contiene un script para despliegue en Hugginface

**Nota** No se suben las imagenes de entrenamiento por protección de datos. Deberas crear los directorios utilizados en cada Script.   

Revisa la aplicación desplegada en HuggingFace: [ergodesk app](https://jeisson46-ergodesk-customvision-mediapipe.hf.space/)
