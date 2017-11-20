# Goal : Read numbers from black and white images

from threading import Thread
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from pprint import pprint
import numpy as np
import matplotlib.pyplot as matplot


print("Import data...")
numbers_db = fetch_mldata('MNIST original')


# Le dataset principal qui contient toutes les images
print(numbers_db.data.shape)
# Le vecteur d'annotations associé au dataset (nombre entre 0 et 9)
print(numbers_db.target.shape)


# Down-sample
print("Down sample...")
sample = np.random.randint(70000, size=5000)
data = numbers_db.data[sample]
target = numbers_db.target[sample]

# Separate learn and test data
xtrain, xtest, ytrain, ytest = train_test_split(data, target, train_size=0.8)

print("Learning for k = 3 ....")
knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(xtrain, ytrain)


error = 1 - knn.score(xtest, ytest)
print('Error for k=3 (nb of neighbours) : %f' % error)


# Trying the same but on all k values from 2 to 15
print("Learning for all k values, 2 to 15")
indexes_k = []
errors = []
for k in range(2, 15):
    print("k = ", k)
    knn = neighbors.KNeighborsClassifier(k)
    errors.append(100*(1 - knn.fit(xtrain, ytrain).score(xtest, ytest)))
    indexes_k.append(k)
pprint(("Error values : ", errors))


def plot_errors():
    matplot.plot(indexes_k, errors, 'o-')
    matplot.show()


t_plot = Thread(target=plot_errors)
t_plot.start()


# On récupère le classifieur le plus performant
best_k_value = min(errors)
index_best_k = errors.index(best_k_value)
best_k = indexes_k[index_best_k]
print("Best k value is ", best_k, ", with an error of ", errors[best_k], "% and will be used now")
knn = neighbors.KNeighborsClassifier(best_k)
knn.fit(xtrain, ytrain)

# On récupère les prédictions sur les données test
predicted = knn.predict(xtrain)

# On redimensionne les données sous forme d'images
images = xtrain.reshape((-1, 28, 28))

# On selectionne un echantillon de 12 images au hasard
select = np.random.randint(images.shape[0], size=12)


# On affiche les images avec la prédiction associée
def plot_predicted():
    for index, value in enumerate(select):
        matplot.subplot(3, 4, index+1)
        matplot.axis('off')
        matplot.imshow(images[value], cmap=matplot.cm.gray_r, interpolation="nearest")
        matplot.title('Predicted: %i' % predicted[value])
    matplot.show()


# Join previous figure before opening another one
t_plot.join()

t_predicted = Thread(target=plot_predicted)
t_predicted.start()
print("Waiting for the join()")
t_predicted.join()
print("Thread joined, ending.")




















