import numpy as np
import math
import random
import pickle

class DecisionNode():
    # Class to represent a single node in a decision tree.
    def __init__(self, left, right, decision_function,class_label=None):
        # Create a node with a left child, right child,decision function and optional class label for leaf nodes
        self.left = left
        self.right = right
        self.decision_function = decision_function
        self.class_label = class_label

    # Return on a label if node is leaf, or pass the decision down to the node's left/right child
    def decide(self, feature):
        if self.class_label is not None:
            return self.class_label

        return self.left.decide(feature) if self.decision_function(feature) else self.right.decide(feature)

'Instrumentaion'
def confusion_matrix(classifier_output, true_labels):
    # output should be [[true_positive, false_negative], [false_positive, true_negative]]
    true_positive = 0
    false_negative = 0
    false_positive = 0
    true_negative = 0

    for index, item in enumerate(classifier_output):
        # classifier predicted yes
        if item == 1:
            if true_labels[index] == 1:
                true_positive += 1
            else:
                false_positive += 1
        # classifier predicted no
        else:
            if true_labels[index] == 0:
                true_negative += 1
            else:
                false_negative += 1

    calculated_confusion_matrix = [[true_positive, false_negative], [false_positive, true_negative]]
    return calculated_confusion_matrix


def precision(classifier_output, true_labels):
    # precision is measured as: true_positive/ (true_positive + false_positive)
    # using  http://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/
    true_positive = 0
    false_positive = 0

    for index, item in enumerate(classifier_output):
        # classifier predicted yes
        if item == 1:
            if true_labels[index] == 1:
                true_positive += 1
            else:
                false_positive += 1

    true_false = true_positive + false_positive
    if true_false == 0:
        return 0
    calculated_precision = float(true_positive) / float(true_false)
    return calculated_precision


def recall(classifier_output, true_labels):
    # recall is measured as: true_positive/ (true_positive + false_negative)
    # using  http://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/
    true_positive = 0
    false_negative = 0

    for index, item in enumerate(classifier_output):
        # classifier predicted yes
        if item == 1:
            if true_labels[index] == 1:
                true_positive += 1
        # classifier predicted no
        elif item == 0:
            if true_labels[index] != 0:
                false_negative += 1

    true_false = true_positive + false_negative
    if true_false == 0:
        return 0

    calculated_recall = float(true_positive) / float(true_false)
    return calculated_recall


def accuracy(classifier_output, true_labels):
    # accuracy is measured as:  correct_classifications / total_number_examples
    total_number_examples = classifier_output.__len__()

    correct_classifications = 0
    for index, item in enumerate(classifier_output):
        if true_labels[index] == item:
            correct_classifications += 1

    if total_number_examples == 0:
        return 0

    calculated_accuracy = float(correct_classifications) / float(total_number_examples)
    return calculated_accuracy

def entropy(class_vector):
    # Note: Classes will be given as either a 0 or a 1.
    if class_vector.__len__() == 0:
        return 0

    vector_size = float(class_vector.__len__())
    ones = float([x for x in class_vector if x == 1].__len__())
    zeros = float([x for x in class_vector if x == 0].__len__())

    probablity_ones = ones / vector_size
    probablity_zeros = zeros / vector_size

    if abs(probablity_ones) > 0:
        calculated_entropy_1 = probablity_ones*np.log2(probablity_ones)
    else:
        calculated_entropy_1 = 0

    if abs(probablity_zeros) > 0:
        calculated_entropy_0 = probablity_zeros*np.log2(probablity_zeros)
    else:
        calculated_entropy_0 = 0

    calculated_entropy = -(calculated_entropy_1 + calculated_entropy_0)
    return calculated_entropy

def information_gain(previous_classes, current_classes ):

    entropy_before = entropy(previous_classes)

    left_class = current_classes[0][0]
    right_class = current_classes[0][1]
    left_class_size = left_class.__len__()
    right_class_size = right_class.__len__()
    size = left_class_size + right_class_size
    probablity_left = float(left_class_size) / float(size)
    probablity_right = float(right_class_size) / float(size)
    left_entropy = entropy(left_class)
    right_entropy = entropy(right_class)
    entropy_after = (probablity_left * left_entropy) + (probablity_right * right_entropy)

    gain = entropy_before - entropy_after

    return gain

def alpha_best_normalized_information_gains(normalized_information_gains):
    if normalized_information_gains.__len__() == 0:
        return 0

    for idx, val in enumerate(normalized_information_gains):
        if math.isnan(val):
            return val

    max_normalized_information_gain = max(normalized_information_gains)
    #print normalized_information_gains
    return max_normalized_information_gain

def split(feature, classifications):
    # split using mean
    mean = np.mean(feature)
    left = []
    right = []

    for index, item in enumerate(feature):
        if item > mean:
            left.append(classifications[index])
        else:
            right.append(classifications[index])

    feature_dict = {1:left, 0:right}
    return feature_dict, mean

def feature_split(feature, classifications, split_value):
    left_feature = []
    left_class = []
    right_feature = []
    right_class = []

    for index, item in enumerate(feature):
        if item > split_value:
            left_feature.append(item)
            left_class.append(classifications[index])
        else:
            right_feature.append(item)
            right_class.append(classifications[index])
    return left_feature, right_feature, left_class, right_class


class DecisionTree():

    def __init__(self, depth_limit=float('inf')):
        self.root = None
        self.depth_limit = depth_limit

    def fit(self, features, classes):
        self.root = self.__build_tree__(features, classes)

    def __build_tree__(self, features, classes, depth=0):
        # Implement the C4.5 algorithm
        if features.size == 0:
            return

        # 1) Check for base cases:
        # a)If all elements of a list are of the same class, return a leaf node with the appropriate class label.
        if len(set(classes)) == 1:
            leaf_node = DecisionNode(None, None, None, classes[0])
            return leaf_node

        # b)If a specified depth limit is reached, return a leaf labeled with the most frequent class.
        depth += 1
        if self.depth_limit == depth:
            ones = [x for x in classes if x == 1].__len__()
            zeros = [x for x in classes if x == 0].__len__()
            if ones > zeros:
                leaf_node = DecisionNode(None, None, None, 1)
            else:
                leaf_node = DecisionNode(None, None, None, 0)
            return leaf_node

        # 2) For each attribute alpha: evaluate the normalized information gain gained by splitting on alpha
        feature_splits = {}
        feature_list = {}
        feature_len = features[0].__len__()

        for i in range(feature_len):
            feature = features[:,i]
            if np.isnan(feature).all():
                continue
            feature_list[i] = feature
            feature_splits[i] = split(feature, classes)

        normalized_information_gains = []
        normalized_information_gains_index_dict = {}
        for attribute, feature in feature_splits.iteritems():
            alpha_information_gain = information_gain(classes,  feature_splits[attribute])
            normalized_information_gains.append(alpha_information_gain)
            normalized_information_gains_index_dict[alpha_information_gain] = attribute

        #3) Let alpha_best be the attribute with the highest normalized information gain
        alpha_best_normalized_information_gain = alpha_best_normalized_information_gains(normalized_information_gains)
        alpha_best = normalized_information_gains_index_dict[alpha_best_normalized_information_gain]

        #4) Create a decision node that splits on alpha_best
        split_value = feature_splits[alpha_best][1]
        alpha_best_feature = feature_list[alpha_best]
        alpha_best_feature_sublists = feature_split(alpha_best_feature, classes, split_value)

        alpha_best_left_feature = alpha_best_feature_sublists[0]
        alpha_best_right_feature = alpha_best_feature_sublists[1]
        alpha_best_left_classes = alpha_best_feature_sublists[2]
        alpha_best_right_classes = alpha_best_feature_sublists[3]

        left_feature_list = []
        alpha_best_left_feature_len = alpha_best_left_feature.__len__()
        for i in range(alpha_best_left_feature_len):
            left = []
            feature = features[i]
            for j in range(feature.__len__()):
                if j != alpha_best:
                    left.append(feature[j])
                else:
                    left.append(alpha_best_left_feature[i])
            left_feature_list.append(left)

        right_feature_list = []
        alpha_best_right_feature_len = alpha_best_right_feature.__len__()
        for i in range(alpha_best_right_feature_len):
            right = []
            feature = features[i + alpha_best_left_feature_len]
            for j in range(feature.__len__()):
                if j != alpha_best:
                    right.append(feature[j])
                else:
                    right.append(alpha_best_right_feature[i])
            right_feature_list.append(right)

        #5) Recur on the sublists obtained by splitting on alpha_best, and add those nodes as children of node
        tree_root = DecisionNode(self.__build_tree__(np.asarray(left_feature_list), alpha_best_left_classes, depth), self.__build_tree__(np.asarray(right_feature_list), alpha_best_right_classes, depth), lambda x: x[0] > split_value)
        return tree_root

    def classify(self, features):
        # Use a fitted tree to classify a list of feature vectors
        # Your output should be a list of class labels (either 0 or 1)
        classifier_output = [self.root.decide(feature) for feature in features]
        return classifier_output

'load data'
def load_csv(data_file_path, class_index=-1):
    handle = open(data_file_path, 'r')
    contents = handle.read()
    handle.close()
    rows = contents.split('\n')
    # Randomize the dataset to avoid bias
    random.shuffle(rows)
    out = np.array([[float(i) for i in r.split(',')] for r in rows if r])
    classes= map(int,  out[:,class_index])
    features = out[:, :class_index]
    return features, classes

def generate_k_folds(dataset, k):
    #this method should return a list of folds,
    k_folds = []
    fold_size = dataset[0].__len__() / k
    j = 0
    pick_len = fold_size

    for i in xrange(k):
        examples_i_fold = dataset[0][j:pick_len]
        classes_i_fold = dataset[1][j:pick_len]

        # where each set is a tuple like (examples, classes)
        training_set = examples_i_fold[0: fold_size - k], classes_i_fold[0: fold_size - k]
        test_set = examples_i_fold[fold_size - k: fold_size], classes_i_fold[fold_size - k: fold_size]

        # where each fold is a tuple like (training_set, test_set)
        fold_i = training_set, test_set
        k_folds.append(fold_i)

        j += fold_size
        pick_len += fold_size

    return k_folds

dataset = load_csv('sample_dataset.csv')
ten_folds = generate_k_folds(dataset, 10)

#on average your accuracy should be higher than 60%.
accuracies = []
precisions = []
recalls = []
confusion = []

for fold in ten_folds:
    train, test = fold
    train_features, train_classes = train
    test_features, test_classes = test

    tree = DecisionTree( )
    tree.fit( train_features, train_classes)
    output = tree.classify(test_features)
    accuracies.append( accuracy(output, test_classes))
    precisions.append( precision(output, test_classes))
    recalls.append( recall(output, test_classes))
    confusion.append( confusion_matrix(output, test_classes))

print '\n'
print 'K-Folds DecisionTree Results'
print '--------------'
print "K-Folds DecisionTree Accuracies:", accuracies
print "K-Folds DecisionTree Precisions:", precisions
print "K-Folds DecisionTree Recalls:", recalls
print "K-Folds DecisionTree Confusion Matrix:", confusion
print "Average of K-Fold DecisionTree Accuracies:", np.mean(accuracies)

'3'
class RandomForest():

    def __init__(self, num_trees, depth_limit, example_subsample_rate, attr_subsample_rate):
        self.trees = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate

    def fit(self, features, classes):
        # implement the above algorithm to build a random forest of decision trees
        example_subsample_size = int(math.ceil(self.example_subsample_rate * features.shape[0]))
        attr_subsample_size = int(math.ceil(self.attr_subsample_rate * features.shape[1]))

        for i in range(self.num_trees):
            # randomized features & associate classes
            features_random_indexes = np.random.randint(0, features.shape[0] - 1, example_subsample_size)
            train_randomized_features_subsample = []
            train_randomized_classes_subsample = []
            for x in np.nditer(features_random_indexes):
                train_randomized_features_subsample.append(features[x])
                train_randomized_classes_subsample.append(classes[x])

            # randomized attr
            randomized_attributes_indexes = random.sample(set([0, 1, 2, 3]), attr_subsample_size)
            for i in range(train_randomized_features_subsample.__len__()):
                randomized_feature = train_randomized_features_subsample[i]
                for j in range(randomized_feature.__len__()):
                    if j not in randomized_attributes_indexes:
                        randomized_feature[j] = None

            tree = DecisionTree(self.depth_limit)
            tree.fit( np.asarray(train_randomized_features_subsample), train_randomized_classes_subsample)
            self.trees.append(tree)

    def classify(self, features):
        # implement classification for a random forest.
        # Your output should be a list of class labels (either 0 or 1)
        classifiers = []
        for tree in self.trees:
            if tree.root is not None:
                classifier_output = [tree.root.decide(feature) for feature in features]
                classifiers.append(classifier_output)

        random_forest_classifiers = np.asarray(classifiers)
        random_forest_classifier_output = []
        for i in range(classifiers[0].__len__()):
            vote_classifier = random_forest_classifiers[:,i]
            win = np.bincount(vote_classifier).argmax()
            random_forest_classifier_output.append(win)

        return random_forest_classifier_output


#As with the DecisionTree, evaluate the performance of your RandomForest on the dataset for part 2.
# on average your accuracy should be higher than 75%.
random_forest_dataset = load_csv('sample_dataset.csv')
random_forest_ten_folds = generate_k_folds(random_forest_dataset, 10)

random_forest_accuracies = []
random_forest_precisions = []
random_forest_recalls = []
random_forest_confusion = []

for random_forest_fold in random_forest_ten_folds:
    random_forest_train, random_forest_test = random_forest_fold
    random_forest_train_features, random_forest_train_classes = random_forest_train
    random_forest_test_features, random_forest_test_classes = random_forest_test

    random_forest = RandomForest(5, 4, 1.0, 1.0)
    random_forest.fit( random_forest_train_features, random_forest_train_classes)
    random_forest_output = random_forest.classify(random_forest_test_features)
    random_forest_accuracies.append( accuracy(random_forest_output, random_forest_test_classes))
    random_forest_precisions.append( precision(random_forest_output, random_forest_test_classes))
    random_forest_recalls.append( recall(random_forest_output, random_forest_test_classes))
    random_forest_confusion.append( confusion_matrix(random_forest_output, random_forest_test_classes))

print '\n'
print 'Random Forest Results'
print '--------------'
print "K-Folds Random Forest Accuracies:", random_forest_accuracies
print "K-Folds Random Forest Precisions:", random_forest_precisions
print "K-Folds Random Forest Recalls:", random_forest_recalls
print "K-Folds Random Forest Confusion Matrix:", random_forest_confusion
print "Average of K-Folds Random Forest Accuracies:", np.mean(random_forest_accuracies)

