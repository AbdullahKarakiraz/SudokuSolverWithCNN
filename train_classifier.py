from Sudokunet import SudokuNet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report


INIT_LR = 1e-3
EPOCHS = 75
BS = 128


print("[INFO] accessing MNIST...")
((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))
testData = testData.reshape((testData.shape[0], 28, 28, 1))

trainData = trainData.astype("float32") / 255.0
testData = testData.astype("float32") / 255.0

le = LabelBinarizer()
trainLabels = le.fit_transform(trainLabels)
testLabels = le.transform(testLabels)

dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.1,
                             rotation_range=10)
dataGen.fit(trainData)

print("[INFO] compiling model...")
opt = Adam(learning_rate=INIT_LR)
model = SudokuNet.Build(width=28,height=28,depth=1,classes=10)
model.compile(loss = "categorical_crossentropy",optimizer=opt,metrics=["accuracy"])

print("[INFO] training network...")
# H = model.fit(trainData,trainLabels,validation_data=(testData,testLabels),batch_size=BS,epochs=EPOCHS,verbose=1)
H = model.fit_generator(dataGen.flow(trainData,trainLabels,batch_size = BS),
                        validation_data=(testData,testLabels),
                        epochs=EPOCHS,verbose=1)
print("[INFO] evaluating network...")
predictions = model.predict(testData)
print(classification_report(testLabels.argmax(axis = 1),predictions.argmax(axis = 1),target_names=[str(x) for x in le.classes_]))

print("[INFO] serializing digit model...")
model.save("digit_classifier8.h5", save_format="h5")