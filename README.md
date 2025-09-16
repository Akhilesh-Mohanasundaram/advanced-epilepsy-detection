# advanced-epilepsy-detection
EEG-driven epilepsy detection platform applying ML/DL for focal vs non-focal classification, tackling data imbalance with SMOTE and enabling reliable, faster medical diagnosis.


#Introduction : 

Epilepsy, a chronic neurological disorder characterized by recurring seizures, affects 
millions of individuals worldwide, profoundly impacting their quality of life. Early and 
accurate detection of epileptic episodes is crucial for effective treatment, monitoring, and 
improving patient outcomes. Electroencephalography (EEG) signals, which capture the 
brain's electrical activity, serve as a critical tool for epilepsy diagnosis. However, the manual 
analysis of EEG data is time-consuming, prone to subjective errors, and often requires 
expert neurologists, making automated and efficient diagnostic methods an essential 
advancement. In this project, titled "Advanced Epilepsy Detection through EEG Signal 
Analysis," we aim to leverage the power of machine learning (ML) and deep learning (DL) to 
automate the classification of EEG signals into focal and non-focal categories. Focal 
seizures originate from specific regions of the brain and often pose unique diagnostic 
challenges. The ability to differentiate between focal and non-focal seizures accurately can 
significantly enhance patient-specific treatment plans. 
The project involves extracting a comprehensive set of features from EEG signals, 
including fractal dimensions (Higuchi and Katz), wavelet-based features, power spectral 
density, spectral entropy, and statistical measures such as mean, variance, skewness, 
and kurtosis. These features capture critical time-domain and frequency-domain 
characteristics of EEG signals, enabling machine learning models to classify them 
effectively. 
To handle the class imbalance commonly found in EEG datasets, we utilize Synthetic 
Minority Oversampling Technique (SMOTE) for creating a balanced training dataset. 
Various machine learning classifiers, including Gradient Boosting, Random Forest, 
XGBoost, and Support Vector Machines (SVM), are trained and evaluated. Furthermore, we 
integrate a Long Short-Term Memory (LSTM)-based deep learning model to exploit 
temporal dependencies in the EEG data. The models are optimized using Randomized 
Search Cross-Validation, and their performances are compared based on accuracy and 
classification metrics. 
This project not only explores state-of-the-art techniques in signal processing and machine 
learning but also addresses real-world challenges in medical diagnosis. By automating 
epilepsy detection, the system holds the potential to reduce diagnostic delays, assist 
neurologists in clinical decision-making, and ultimately improve the quality of life for 
individuals living with epilepsy.
