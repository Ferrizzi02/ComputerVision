from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datatreatment.datasets.simpledatasetloader import SimpleDataSetLoader
from datatreatment.preprocessing.simplepreprocessor import SimplePreprocessor
from imutils import paths
import argparse

#--dataset
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1,help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1, help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())

print("[INFO] cargando imagenes...")
imagePaths = list(paths.list_images(args["dataset"]))

#preprocessor
sp = SimplePreprocessor(32,32)
sdl = SimpleDataSetLoader(preprocessors=[sp])
(data,labels) = sdl.load(imagePaths, 500)

data = data.reshape((data.shape[0], 3072)) #np array
#tipo psd 3000 *32 *32* 3 => 3000, 3072
print("[INFO] tamanho matriz: {:.1f}MB".format(data.nbytes / (1024 * 1000.0)))
# tamanho do array para MB

#discretização
le = LabelEncoder()
labels = le.fit_transform(labels)

#divisão do dataset
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

#KNN
model = KNeighborsClassifier(n_neighbors=args["neighbors"], n_jobs=args["jobs"])
model.fit(trainX, trainY)
print(classification_report(testY, model.predict(testX), target_names=le.classes_))
