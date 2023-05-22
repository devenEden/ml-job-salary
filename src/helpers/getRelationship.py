
import matplotlib.pyplot as plt

def getRelationship (data, target, dep):
    plt.scatter(data[dep], data[target])
    plt.xlabel(dep)
    plt.ylabel(target)
    plt.title("Relationship between " + dep + " and " + target)
    plt.show()