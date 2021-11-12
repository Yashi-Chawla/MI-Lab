import numpy as np


class KNN:
    """
    K Nearest Neighbours model
    Args:
        k_neigh: Number of neighbours to take for prediction
        weighted: Boolean flag to indicate if the nieghbours contribution
                  is weighted as an inverse of the distance measure
        p: Parameter of Minkowski distance
    """

    def __init__(self, k_neigh, weighted=False, p=2):

        self.weighted = weighted
        self.k_neigh = k_neigh
        self.p = p

    def fit(self, data, target):
        """
        Fit the model to the training dataset.
        Args:
            data: M x D Matrix( M data points with D attributes each)(float)
            target: Vector of length M (Target class for all the data points as int)
        Returns:
            The object itself
        """

        self.data = data
        self.target = target.astype(np.int64)

        return self
   
    def find_distance(self, x):
        """
        Find the Minkowski distance to all the points in the train dataset x
        Args:
            x: N x D Matrix (N inputs with D attributes each)(float)
        Returns:
            Distance between each input to every data point in the train dataset
            (N x M) Matrix (N Number of inputs, M number of samples in the train dataset)(float)
        """
        # TODO
        distances=[[] for i in range(len(x))]
        for rownum in range(len(x)):
            row0=x[rownum]
            for row1 in self.data:
                dist=0.0
                dist=pow(sum(pow(abs(row1-row0),self.p)),1/self.p)
                distances[rownum].append(dist)
        return(distances)
        
    def k_neighbours(self, x):
        """
        Find K nearest neighbours of each point in train dataset x
        Note that the point itself is not to be included in the set of k Nearest Neighbours
        Args:
            x: N x D Matrix( N inputs with D attributes each)(float)
        Returns:
            k nearest neighbours as a list of (neigh_dists, idx_of_neigh)
            neigh_dists -> N x k Matrix(float) - Dist of all input points to its k closest neighbours.
            idx_of_neigh -> N x k Matrix(int) - The (row index in the dataset) of the k closest neighbours of each input

            Note that each row of both neigh_dists and idx_of_neigh must be SORTED in increasing order of distance
        """
        # TODO
        neigh_dists=[[] for i in range(len(x))]
        idx_of_neigh=[[] for i in range(len(x))]
        distances=KNN.find_distance(self,x)
        dist=np.argsort(distances)
        for i in range(len(dist)):
            row=dist[i]
            for j in range(0,self.k_neigh):
                idx_of_neigh[i].append(row[j])

        for j in range(len(idx_of_neigh)):
            row=idx_of_neigh[j]
            for i in row:
                neigh_dists[j].append(distances[j][i])
        return (neigh_dists,idx_of_neigh)

    def make_weights(self,distances):
        results=[0 for i in range(self.k_neigh)]
        sum=0.0
        for i in range(self.k_neigh):
            results[i]+=1.0/(distances[i]+pow(10,-9))
            sum+=results[i]
        results=[results[i]/sum for i in range(len(results))]
        return results

    def predict(self, x):
        """
        Predict the target value of the inputs.s
        Args:
            x: N x D Matrix( N inputs with D attributes each)(float)
        Returns:
            pred: Vector of length N (Predicted target value for each input)(int)
        """
        # TODO
        neigh_dists,idx_of_neigh=KNN.k_neighbours(self,x)

        if self.weighted==False:
            pred=list()
            output_values=[[]for i in range(len(idx_of_neigh))]
            for i in range(len(idx_of_neigh)):
                row=idx_of_neigh[i]
                for j in row:
                    output_values[i].append(self.target[j])
            for i in output_values:
                prediction = max(set(i), key=i.count)
                pred.append(prediction)

        else:
            pred=[[] for i in range(len(neigh_dists))]
            dists=neigh_dists
            weights=list()
            for i in range(len(dists)):
                row=dists[i]
                weights.append(KNN.make_weights(self,row))
            for i in range(len(idx_of_neigh)):
                classes=dict()
                row=idx_of_neigh[i]
                for j in range(len(row)):
                    k=row[j]
                    pred_class=self.target[k]
                    if pred_class not in classes.keys():
                        classes[pred_class]=weights[i][j]
                    else:
                        classes[pred_class]+=weights[i][j]
                max_pred=max(classes,key=classes.get)
                pred[i].append(max_pred)
            pred = [item for sublist in pred for item in sublist]
        return pred

    def accuracy_metric(self,actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct /len(actual)* 100.0
    
    def evaluate(self, x, y):
        """
        Evaluate Model on test data using 
            classification: accuracy metric
        Args:
            x: Test data (N x D) matrix(float)
            y: True target of test data(int)
        Returns:
            accuracy : (float.)
        """
        # TODO
        predicted=KNN.predict(self,x)
        accuracy=KNN.accuracy_metric(self,y,predicted)
        return accuracy