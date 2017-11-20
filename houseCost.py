# ######################################################################################################################
# ############################                           libraries                          ############################
# ######################################################################################################################
import numpy as np
import pandas
import matplotlib.pyplot as mat_plot
from pprint import pprint
from collections import Counter


# ######################################################################################################################
# ############################                Dataset loading and preparation               ############################
# ######################################################################################################################

# Pandas(Index=, price=, surface=, arrondissement=)
house_data = pandas.read_csv("house_data_extended.csv", na_filter=False)


# Exclude rows with missing data
# excluded_data = {'price': [], 'surface': [], 'arrondissement': []}
indexes_to_exclude = []
for i in range(len(house_data['arrondissement'])):
    # print(house_data['arrondissement'][i])
    if house_data['arrondissement'][i] in (None, ""):
        print(i)
        indexes_to_exclude += [i]


# Building the excluded data (why?)
excluded_data = pandas.DataFrame(
    {'price': np.array([house_data['price'][i] for i in indexes_to_exclude]),
     'surface': np.array([house_data['surface'][i] for i in indexes_to_exclude]),
     'arrondissement': np.array([house_data['arrondissement'][i] for i in indexes_to_exclude])})


print("len of the data", len(house_data))
house_data.drop(house_data.index[indexes_to_exclude], inplace=True)
print("len of the data", len(house_data), "\n")

# Convert to numeric
house_data["surface"] = pandas.to_numeric(house_data["surface"])
house_data["arrondissement"] = pandas.to_numeric(house_data["arrondissement"])


# ######################################################################################################################
# Checking results
print(" ******  Pandas tables ******")
pprint("Data sets : EXCLUDED Data")
pprint(excluded_data)
pprint("Data sets : HOUSE Data")
pprint(house_data)
print("\n")

print("Test des types ")
print("ex value of surface : ", house_data["surface"][1], "de type : ", type(house_data["surface"][1]))
print("ex value of price : ", house_data["price"][1], "de type : ", type(house_data["price"][1]))
print("ex value of arrondissement : ", house_data["arrondissement"][1], "de type : ", type(house_data["arrondissement"][1]))
print("\n")
# OK DATA CLEAN

# ######################################################################################################################
#
#  Try to sort into categories
#
house_data = house_data.sort_values(by=["arrondissement", "surface"])
pprint("Data sets : House Data SORTED")
pprint(house_data)


# Pandas(Index=, price=, surface=, arrondissement=)
arrondissements = Counter(house_data['arrondissement']).keys()
print("Counter des arrondissements : ", arrondissements)

# data_per_arrond = {i: {'price': [], 'surface': []} for i in sorted(arrondissements)}
# for row in house_data.itertuples():
#     # print(row)
#     data_per_arrond[row[3]]['price'].append(row[1])
#     data_per_arrond[row[3]]['surface'].append(row[2])

# print("data per arrondissement")
# pprint(data_per_arrond)
# print("done")



fig, ax = mat_plot.subplots()

for arrond in arrondissements:
    ax.scatter(data_per_arrond[arrond]['price'], data_per_arrond[arrond]['surface'])
    ax.xlabel("Surface")
    ax.ylabel("Price")
mat_plot.show()


mat_plot.scatter(house_data['surface'], house_data['price'], c=house_data['arrondissement'])
mat_plot.xlabel("Surface")
mat_plot.ylabel("Price")
mat_plot.legend(["Arrond. " + str(i) for i in range(1, 5)])
mat_plot.show()




# On décompose le dataset et on le transforme en matrices pour pouvoir effectuer notre calcul
X = np.matrix([np.ones(house_data.shape[0]), house_data['surface'].as_matrix()]).T
y = np.matrix(house_data['loyer']).T

# On effectue le calcul exact du paramètre theta
theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

print(theta)


mat_plot.xlabel('Surface')
mat_plot.ylabel('Loyer')

mat_plot.plot(house_data['surface'], house_data['loyer'], 'ro', markersize=4)

# On affiche la droite entre 0 et 250
mat_plot.plot([0,250], [theta.item(0),theta.item(0) + 250 * theta.item(1)], linestyle='--', c='#000000')


print("Loyer sera d'environ : {}€".format(theta.item(0) + theta.item(1) * int(input("Taille de l'appart : "))))

mat_plot.show()

data_size = len(house_data)
sample = np.random.randint(data_size, size=data_size*0.1)
sampled_data = data[sample]

