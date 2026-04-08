from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from preprocess import EEGData

proc = EEGData()
data_path = "/Users/anusha/flappy_bird_data/2.23_allyson_squeeze/2_0002_raw.edf"
X, y = proc.load_and_process(data_path)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

clf = SVC(kernel='rbf')
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(f"Classification Accuracy: {accuracy:.2%}")