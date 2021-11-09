import pandas as pd
import math
import statistics

from sklearn import datasets

def get_knn_class(k, df, test_point):
    """
    Get class of the test point using the k nearest neighbors algorithm
    """
    distances = []

    for train_point in df.itertuples(index=False):

        # Calculate distance
        sum = 0
        for i in range(4):
            sum += pow(test_point[i] - train_point[i], 2)

        distance = math.sqrt(sum)

        distances.append((distance, train_point))
    
    distances.sort()
    classes = []

    for i in range(k):
        classes.append(distances[i][1].iris_class)
    
    return statistics.mode(classes)

def get_data():
    """
    Load data from sklearn datasets and format it into a formatted dictionary for the DataFrame Constructor
    """

    iris_data = datasets.load_iris()
    
    attributes = ("sepal_length", "sepal_width", "pedal_length", "pedal_width", "iris_class")
    formatted_data = {}

    for attribute in attributes:
        formatted_data[attribute] = []
    
    data = iris_data["data"]
    target = iris_data["target"]

    for i in range(len(data)):
        for j in range(4):
            formatted_data[attributes[j]].append(data[i][j])
        
        formatted_data["iris_class"].append(target[i])

    return formatted_data
    
    
if __name__ == "__main__":

    # Load data into a pandas dataframe
    df = pd.DataFrame(get_data())
    
    # Take 20% of data for testing
    test_df = df.sample(n=None, frac=0.2)
    train_df = df.drop(test_df.index)

    # Test
    tested = 0
    correct = 0
    for test_point in test_df.itertuples(index=False):

        classified = get_knn_class(7, train_df, test_point)

        print(f"{test_point.iris_class} Classified as: {classified}")

        tested += 1
        if test_point.iris_class == classified:
            correct += 1

    print(f"Accuracy: {correct/tested}")


    
    
