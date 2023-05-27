


import numpy as np

depth_value = 0
minimum_split_value = 2

class tree_State():
    def __init__(curr, let_feature_index=None, let_threshold=None, let_left=None, let_right=None, let_info_gain=None, let_value=None):
        #  constructor

        # for (Internal) decision node
        curr.feature_index = let_feature_index
        curr.threshold = let_threshold
        curr.left = let_left
        curr.right = let_right
        curr.info_gain = let_info_gain

        # for leaf node
        curr.value = let_value


class make_decision_tree():
    def __init__(curr, minimum_sample_splits=minimum_split_value, maxim_depth=depth_value):
        #  constructor

        # initialize the root of the tree
        curr.root = None

        # stopping conditions
        curr.minimum_sample_splits = minimum_sample_splits
        curr.maxim_depth = maxim_depth

     #function to build the tree
    def build_tree(self, dataset, be_the_depth=0):
       

        X_data, Y_data = dataset[:, :-1], dataset[:, -1]
        number_of_samples, number_of_features = np.shape(X_data)

        # spliting until breaking conditions are met
        if number_of_samples >= self.minimum_sample_splits and be_the_depth <= self.maxim_depth:
            # finding out the best split
            final_split = self.get_best_split(
                dataset, number_of_samples, number_of_features)
            # checking if information gain is positive using if loop
            if final_split["informa_gain"] > 0:
                # finding left subtree
                left_subtree = self.build_tree(
                    final_split["left_side_data"], be_the_depth+1)
                # finding right subtree
                right_subtree = self.build_tree(
                    final_split["right_side_data"], be_the_depth+1)
                # returning decision node after the splitting
                return tree_State(final_split["feature_index"], final_split["threshold"],
                             left_subtree, right_subtree, final_split["informa_gain"])

        # computing the leaf node
        got_leaf_value = self.calculate_leaf(Y_data)
        # returning leaf node
        return tree_State(let_value=got_leaf_value)

         #  function to find the best split
    def get_best_split(self, dataset, number_of_samples, number_of_features):
       
        # empty dictionary to store the best split
        final_split = {}
        max_info_gain = -float("inf")

        # loop over all the features
        for feature_index in range(number_of_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            # loop for over all the feature values present in the data
            for threshold in possible_thresholds:
                # get the current split
                left_split_dataset, right_split_dataset = self.splitting_the_data(
                    dataset, feature_index, threshold)
                # checking if childs are not null
                if len(left_split_dataset) > 0 and len(right_split_dataset) > 0:
                    y, left_y, right_y = dataset[:, -
                                                 1], left_split_dataset[:, -1], right_split_dataset[:, -1]
                    # computing the information gain
                    curr_info_gain = self.inform_gain(
                        y, left_y, right_y)
                    # updating the best split if needed
                    if curr_info_gain > max_info_gain:
                        final_split["feature_index"] = feature_index
                        final_split["threshold"] = threshold
                        final_split["left_side_data"] = left_split_dataset
                        final_split["right_side_data"] = right_split_dataset
                        final_split["informa_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain

        # returning the best split
        return final_split

    def splitting_the_data(self, res_dataset, f_index, f_thres):
        #  function to split the dataset

        left_split_dataset = np.array(
            [z for z in res_dataset if z[f_index] <= f_thres])
        right_split_dataset = np.array(
            [z for z in res_dataset if z[f_index] > f_thres])
        return left_split_dataset, right_split_dataset

    def inform_gain(self, parent_side, leftside_child, rightside_child):
        #  function to compute information gain

        leftside_weight = len(leftside_child) / len(parent_side)
        rightside_weight = len(rightside_child) / len(parent_side)
        resultant_gain = self.get_entropy(
            parent_side) - (leftside_weight*self.get_entropy(leftside_child) + rightside_weight*self.get_entropy(rightside_child))
        return resultant_gain

    def get_entropy(self, ol):
        #  function to find the entropy

        class_labels_value = np.unique(ol)
        let_entropy = 0
        for label in class_labels_value:
            p_label = len(ol[ol == label]) / len(ol)
            let_entropy += -p_label * np.log2(p_label)
        return let_entropy

    def calculate_leaf(self, leaf):
        #  function to calculate leaf node

        leaf = list(leaf)
        return max(leaf, key=leaf.count)


    def fiting_tree(self, p, q):
        #  function to train the tree using this function

        tree_dataset = np.concatenate((p, q), axis=1)
        self.root = self.build_tree(tree_dataset)

    def do_predictions(self, u, given_tree):
        #  function to predict a single data point

        if given_tree.value != None:
            return given_tree.value
        featur_val = u[given_tree.feature_index]
        if featur_val <= given_tree.threshold:
            return self.do_predictions(u, given_tree.left)
        else:
            return self.do_predictions(u, given_tree.right)


def main():
    print('STARTING THE QUESTION Q1_AB\n')

    # Take the dataset
    A_train , A_test , B_train , B_test = [], [], [], []

    
    
    
# Reading the train data path 
    train_data_path ='datasets/Q1_train.txt'
    read_data_file = open(train_data_path)
    required_line = read_data_file.readline() 
    required_train_data = []
    while required_line:
        required_line = required_line.rstrip()
        required_train_data.append(required_line)
        required_line = read_data_file.readline()

    output = []

    find_train_data = []
    for info in required_train_data:
        empty_str = ""
        for val in info:
            if (val == '(' or val == ')' or val == ' '):
                continue
            else:
                empty_str += val
        k = empty_str.split(',')
        find_train_data.append(k)

    for data_row in find_train_data:
        temp_arr = []
        temp_arr.append(float(data_row[0]))
        temp_arr.append(float(data_row[1]))
        temp_arr.append(int(data_row[2]))
        
        temp_arr2 = []
        temp_arr2.append(temp_arr)
        [[temp_arr],]
        if (data_row[3] == 'M'):
            temp_arr2.append([1])
        else:
            temp_arr2.append([0])

        output.append(temp_arr2)
    given_train_data =output
    
    length_of_train = len(given_train_data)
    for i in given_train_data:
        A_train.append(i[0])
        B_train.append(i[1])


    
#reading test data path    
    test_data_path = 'datasets/Q1_test.txt'
    test_data_file = open(test_data_path)
    data_line = test_data_file.readline() 
    test_required_data = []
    while data_line:
        data_line = data_line.rstrip()
        test_required_data.append(data_line)
        data_line = test_data_file.readline()

    output2 = []

    test_data_req = []
    for row_ in test_required_data:
        str_ = ""
        for i in row_:
            if (i == '(' or i == ')' or i == ' '):
                continue
            else:
                str_ += i
        m = str_.split(',')
        test_data_req.append(m)

    for rows_ in test_data_req:
        temp_arr3 = []
        temp_arr3.append(float(rows_[0]))
        temp_arr3.append(float(rows_[1]))
        temp_arr3.append(int(rows_[2]))
        
        temp_arr4 = []
        temp_arr4.append(temp_arr3)
        [[temp_arr3],]
        if (rows_[3] == 'M'):
            temp_arr4.append([1])
        else:
            temp_arr4.append([0])

        output2.append(temp_arr4)
    given_test_data = output2


    length_of_test = len(given_test_data)
    for erow in given_test_data:
        A_test.append(erow[0])
        B_test.append(erow[1])

    no_of_split = 3
    no_of_dep = 1

    for i in range(5):
        # calling the classifier
        classifier = make_decision_tree(
            minimum_sample_splits=no_of_split, maxim_depth=no_of_dep)
        classifier.fiting_tree(A_train, B_train)

        print("DEPTH = ", no_of_dep)
        train_predictions =[]
        for x in A_train: 
            predictions_got = classifier.do_predictions(x, classifier.root)
            train_predictions.append(predictions_got)
        B_train_pred = train_predictions
        
        #calculating accuracy
        op_accuracy = []
        for i in range(len(B_train_pred)):
            if B_train_pred[i] == B_train[i]:
                op_accuracy.append(1)
            else:
                op_accuracy.append(0)

        train_dataset_accuracy= np.round(np.mean(op_accuracy), decimals=2)


        test_predictions =[]
        for x in A_test: 
            predictions_got = classifier.do_predictions(x, classifier.root)
            test_predictions.append(predictions_got)
        B_test_pred = test_predictions
        
        
        res_accuracy = []
        for i in range(len(B_test_pred)):
            if B_test_pred[i] == B_test[i]:
                res_accuracy.append(1)
            else:
                res_accuracy.append(0)

        test_data_accuracy= np.round(np.mean(res_accuracy), decimals=2)
        print("Accuracy for | Train data set = ", train_dataset_accuracy,
              " | Test data set=  ", test_data_accuracy)
        no_of_dep = no_of_dep + 1


    print('\nENDING THE QUESTION Q1_AB\n')


if __name__ == "__main__":
    main()






