# ENGG680 Group 9 Final Project
Michael Shi,        UCID: 10174675
Sheinn Min Thu,     UCID: 30200456
Hsu Hsu Sandi,      UCID: 30262965
Hlayn Toe Wai,      UCID: 30266673
Gopika Menon,       UCID: 30046245
Tara Rezaei,        UCID: 30263674
Elnaz Gholizadeh,   UCID: 30264047
Sahar Sedarat,      UCID: 30234160

(*Final Report contains certificate of work*)

## Overview:
We have created a Deep Neural Network to predict traffic incidents in Calgary.
Our predictive model is trained on historical traffic incident data from the City of Calgary to learn patterns between temporal features, spatial features, and incident occurrences. During testing, the model predicts whether a given data point represents an incident or a non-incident. After training, it generates probabilities for incidents across Calgary by applying user-provided time inputs (day, month, hour) to the dataset. These inputs allow the model to predict the likelihood of incidents at various locations. The results are visualized as an interactive heatmap, providing a clear view of incident probabilities across Calgary.

## Instructions:
**Note:** Performance Results and Heatmap Outputs may vary from the final report as the parameters may have been changed/reverted as we continue to experiment with different settings to improve performance. The generated random negative samples may also be different, as well as the random subset selected from the total dataset. This will result in different outputs and performance. 

**Order of file execution:**
- data_processing.py
- negative_sampling.py
- feature_engineering.py
- model_training.py
- generate_map.py

## Required Libraries:
- numpy: pip install numpy==1.25.2
- pandas: pip install pandas==2.2.3
- scikit-learn: pip install scikit-learn==1.3.1
- tensorflow: pip install tensorflow==2.11.0
- matplotlib: pip install matplotlib==3.8.0
- folium: pip install folium==0.14.0
- requests: pip install requests==2.31.0
- Geohash: pip install Geohash==1.0
- seaborn (optional, for improved data visualizations): pip install seaborn==0.12.2