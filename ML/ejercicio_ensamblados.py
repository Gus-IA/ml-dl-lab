from sklearn.datasets import fetch_openml
import numpy as np

from sklearn.datasets import fetch_openml
import numpy as np

mnist = fetch_openml("mnist_784", version=1)
mnist.target = mnist.target.astype(np.uint8)

from sklearn.model_selection import train_test_split

X_train_val, X_test, y_train_val, y_test = train_test_split(
    mnist.data, mnist.target, test_size=1000, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=1000, random_state=42
)

print(X_train.shape, X_test.shape, X_val.shape)

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC

random_forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
extra_trees_clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
svm_clf = LinearSVC(random_state=42)

estimators = [random_forest_clf, extra_trees_clf, svm_clf]
for estimator in estimators:
    print("Training the", estimator)
    estimator.fit(X_train, y_train)

print([estimator.score(X_val, y_val) for estimator in estimators])

from sklearn.ensemble import VotingClassifier

named_estimators = [
    ("random_forest_clf", random_forest_clf),
    ("extra_trees_clf", extra_trees_clf),
    ("svm_clf", svm_clf),
]

voting_clf = VotingClassifier(named_estimators)

voting_clf.fit(X_train, y_train)

voting_clf.score(X_val, y_val)

voting_clf.voting = "soft"
voting_clf.score(X_test, y_test)

voting_clf.voting = "hard"
voting_clf.score(X_test, y_test)

X_val_predictions = np.empty((len(X_val), len(estimators)), dtype=np.float32)

for index, estimator in enumerate(estimators):
    X_val_predictions[:, index] = estimator.predict(X_val)

print(X_val_predictions)

rnd_forest_blender = RandomForestClassifier(
    n_estimators=200, oob_score=True, random_state=42
)
rnd_forest_blender.fit(X_val_predictions, y_val)

rnd_forest_blender.oob_score_


X_test_predictions = np.empty((len(X_test), len(estimators)), dtype=np.float32)

for index, estimator in enumerate(estimators):
    X_test_predictions[:, index] = estimator.predict(X_test)

y_pred = rnd_forest_blender.predict(X_test_predictions)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))
