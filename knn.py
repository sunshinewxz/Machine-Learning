import numpy as np

class KNN():
    # train_data: m*f; test_data: n*f
    def knnClassify(self, train_data, train_label, test_data, k):
        distance_matrix = self.getDistances(train_data, test_data)
        neighbours = self.getNeighbours(distance_matrix, k)
        prediction = self.getResponse(neighbours, train_label)
        return prediction

    def getDistances(self, train_data, test_data):
        # get a test sample
        distance_matrix = []
        index = [i for i in range(len(train_data))]
        for test in test_data:
            distance = 0.5 * np.sqrt(np.sum(((train_data - test) ** 2), axis=1))
            distance = np.vstack((index, distance))
            distance_matrix.append(distance.T)
        return distance_matrix

    def getNeighbours(self, distance_matrix, k):
        neighbours = []
        for dis in distance_matrix:
            dis = dis[dis[:, 1].argsort()]
            neighbour_temp = dis[0:k, 0]
            neighbours.append(neighbour_temp)
        return neighbours

    def getResponse(self, neighbours, train_label):
        pre_class = []
        for test_n in neighbours:
            class_vote = {}
            for n in test_n:
                if train_label[int(n)] in class_vote:
                    class_vote[train_label[int(n)]] += 1
                else:
                    class_vote[train_label[int(n)]] = 1
            class_vote = sorted(class_vote.items(), key=lambda d: d[1], reverse=True)
            pre_class.append(class_vote[0][0])
        return pre_class

knn = KNN()
train_data = np.array([[10,20,30],[5,3,1],[1,0,2],[0.5,1,3],[1,1.5,3]])
train_label = np.array([3,2,1,1,1])
test_data = np.array([[20,30,50],[10,9,1],[0.3,0.1,1]])
k = 3
prediction = knn.knnClassify(train_data, train_label, test_data, k)
