from imblearn.over_sampling import SMOTE
from collections import Counter

def apply_smote(X_train_processed, y_train_processed):
    print('Class Distribution Before SMOTE')
    print(Counter(y_train_processed))
    smote = SMOTE(random_state=42)
    x_train_balanced, y_train_balanced = smote.fit_resample(X_train_processed, y_train_processed)
    print('Class Distribution After SMOTE')
    print(Counter(y_train_balanced))

    print("------Smote applied------")
    print("13. SMOTE Applied")

    return x_train_balanced, y_train_balanced