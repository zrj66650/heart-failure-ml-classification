import math


def classify_knn(training_filename, testing_filename, k):
    # read data
    training_data = []
    with open(training_filename, 'r') as f:
        for line in f:
            line = line.strip()
            parts = line.split(',')
            # all parameters of the patient
            features = parts[:-1]
            # the final result
            label = parts[-1]
            training_data.append((features, label))

    testing_data = []
    with open(testing_filename, 'r') as f:
        for line in f:
            line = line.strip()
            features = line.split(',')
            testing_data.append(features)

    results = []

    for test in testing_data:
        distances = []

        # calculate distance for training data
        for i in range(len(training_data)):
            train_features = training_data[i][0]
            train_label = training_data[i][1]

            distance = 0
            for j in range(len(test)):
                if test[j] != train_features[j]:
                    distance += 1
            
            # prioritize distance firstly, then consider numbering
            distances.append((distance, i, train_label))

        # sort
        distances.sort()

        # take the first k neighbors
        k_nearest = distances[:k]

        # count number
        count_died = 0
        count_survived = 0

        for item in k_nearest:
            label = item[2]
            if label == "died":
                count_died += 1
            else:
                count_survived += 1

        # conclusion
        if count_died > count_survived:
            results.append("died")
        elif count_survived > count_died:
            results.append("survived")
        else:
            # tie
            results.append("died")

    return results


def classify_dt(training_filename, testing_filename):

    # read data
    def load_training(filename):
        data = []
        with open(filename, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                features = parts[:-1]
                label = parts[-1]
                data.append((features, label))
        return data

    def load_testing(filename):
        data = []
        with open(filename, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                data.append(parts)
        return data

    training_data = load_training(training_filename)
    testing_data = load_testing(testing_filename)

    # calculate entropy
    def entropy(data):
        count_died = 0
        count_survived = 0

        for item in data:
            if item[1] == "died":
                count_died += 1
            else:
                count_survived += 1

        total = count_died + count_survived

        if total == 0:
            return 0

        ent = 0

        for count in [count_died, count_survived]:
            if count != 0:
                p = count / total
                ent -= p * math.log2(p)

        return ent

    # calculate information gain
    def information_gain(data, attr_index):
        total_entropy = entropy(data)

        # group by attribute value
        subsets = {}
        for item in data:
            key = item[0][attr_index]
            if key not in subsets:
                subsets[key] = []
            subsets[key].append(item)

        # calculate conditional entropy
        weighted_entropy = 0
        total = len(data)

        for subset in subsets.values():
            weighted_entropy += (len(subset) / total) * entropy(subset)

        # ig
        return total_entropy - weighted_entropy

    # judgment result
    def majority_class(data):
        count_died = 0
        count_survived = 0

        for item in data:
            if item[1] == "died":
                count_died += 1
            else:
                count_survived += 1

        if count_died > count_survived:
            return "died"
        elif count_survived > count_died:
            return "survived"
        # tie
        else:
            return "died"

    # build tree
    def build_tree(data, attributes):
        
        # stopping condition
        # check class
        first_label = data[0][1]
        all_same = True
        for item in data:
            if item[1] != first_label:
                all_same = False
                break
        if all_same:
            return first_label

        # check attributes
        def all_same_attributes(data):
            first = data[0][0]
            for item in data:
                if item[0] != first:
                    return False
            return True

        if all_same_attributes(data):
            return majority_class(data)
        
        if len(attributes) == 0:
            return majority_class(data) 

        # find best attribute
        best_attr = attributes[0]
        best_gain = information_gain(data, best_attr)

        for attr in attributes:
            gain = information_gain(data, attr)
            if gain > best_gain:
                best_gain = gain
                best_attr = attr

        tree = {
            "attribute": best_attr,
            "children": {},
            "majority": majority_class(data)
        }

        # get value of best_attr
        values = []
        for item in data:
            val = item[0][best_attr]
            if val not in values:
                values.append(val)

        for val in values:
            subset = []
            for item in data:
                if item[0][best_attr] == val:
                    subset.append(item)
            
            # remove used features
            new_attributes = []
            for a in attributes:
                if a != best_attr:
                    new_attributes.append(a)
            
            # dealing with children
            if len(subset) == 0:
                tree["children"][val] = majority_class(data)
            else:
                tree["children"][val] = build_tree(subset, new_attributes)

        return tree

    # classification for testing
    def classify(tree, example):
        if type(tree) == str:
            return tree

        attr = tree["attribute"]
        value = example[attr]

        if value in tree["children"]:
            return classify(tree["children"][value], example)
        else:
            return tree["majority"]

    # print tree
    def print_tree(tree, depth=0):
        if type(tree) == str:
            print("  " * depth + "-> " + tree)
            return

        attr = tree["attribute"]
        for val in tree["children"]:
            print("  " * depth + f"[attr {attr} = {val}]")
            print_tree(tree["children"][val], depth + 1)

    # get attributes and construct trees
    num_attributes = len(training_data[0][0])
    attributes = []
    for i in range(num_attributes):
        attributes.append(i)

    tree = build_tree(training_data, attributes)

    # print_tree(default not to print first)

    # processing testing_data
    results = []
    for example in testing_data:
        label = classify(tree, example)
        results.append(label)

    return results


def classify_knnplus(training_filename, testing_filename, k):
    # read data
    training_data = []
    with open(training_filename, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            # all parameters of the patient
            features = parts[:-1]
            # the final result
            label = parts[-1]
            training_data.append((features, label))

    testing_data = []
    with open(testing_filename, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            testing_data.append(parts)

    results = []

    for test in testing_data:
        distances = []

        # calculate distance for training data
        for i in range(len(training_data)):
            train_features = training_data[i][0]
            train_label = training_data[i][1]

            distance = 0
            for j in range(len(test)):
                if test[j] != train_features[j]:
                    distance += 1

            # prioritize distance firstly, then consider numbering
            distances.append((distance, i, train_label))

        # sort
        distances.sort()

        # take the first k neighbors
        k_nearest = distances[:k]

        # ***use distance related weights = 1/(distance+1)***
        score_died = 0
        score_survived = 0

        for item in k_nearest:
            distance = item[0]
            label = item[2]

            weight = 1 / (distance + 1)

            if label == "died":
                score_died += weight
            else:
                score_survived += weight

        # conclusion
        if score_died > score_survived:
            results.append("died")
        elif score_survived > score_died:
            results.append("survived")
        else:
            # tie
            results.append("died")

    return results


def classify_dtplus(training_filename, testing_filename):

    # read data
    def load_training(filename):
        data = []
        with open(filename, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                features = parts[:-1]
                label = parts[-1]
                data.append((features, label))
        return data

    def load_testing(filename):
        data = []
        with open(filename, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                data.append(parts)
        return data

    training_data = load_training(training_filename)
    testing_data = load_testing(testing_filename)

    # calculate entropy
    def entropy(data):
        count_died = 0
        count_survived = 0

        for item in data:
            if item[1] == "died":
                count_died += 1
            else:
                count_survived += 1

        total = count_died + count_survived

        if total == 0:
            return 0

        ent = 0

        for count in [count_died, count_survived]:
            if count != 0:
                p = count / total
                ent -= p * math.log2(p)

        return ent

    # calculate information gain
    def information_gain(data, attr_index):
        total_entropy = entropy(data)

        # group by attribute value
        subsets = {}
        for item in data:
            key = item[0][attr_index]
            if key not in subsets:
                subsets[key] = []
            subsets[key].append(item)

        # calculate conditional entropy
        weighted_entropy = 0
        total = len(data)

        for subset in subsets.values():
            weighted_entropy += (len(subset) / total) * entropy(subset)

        # ig
        return total_entropy - weighted_entropy

    # ***calculate split information***
    def split_information(data, attr_index):
        subsets = {}

        for item in data:
            key = item[0][attr_index]
            if key not in subsets:
                subsets[key] = []
            subsets[key].append(item)

        total = len(data)
        si = 0

        for subset in subsets.values():
            p = len(subset) / total
            if p != 0:
                si -= p * math.log2(p)

        return si


    # judgment result
    def majority_class(data):
        count_died = 0
        count_survived = 0

        for item in data:
            if item[1] == "died":
                count_died += 1
            else:
                count_survived += 1

        if count_died > count_survived:
            return "died"
        elif count_survived > count_died:
            return "survived"
        # tie
        else:
            return "died"

    # build tree
    def build_tree(data, attributes):
        
        # stopping condition
        # check class
        first_label = data[0][1]
        all_same = True
        for item in data:
            if item[1] != first_label:
                all_same = False
                break
        if all_same:
            return first_label

        # check attributes
        def all_same_attributes(data):
            first = data[0][0]
            for item in data:
                if item[0] != first:
                    return False
            return True

        if all_same_attributes(data):
            return majority_class(data)
        
        if len(attributes) == 0:
            return majority_class(data) 

        # ***calculate and use split information***
        # (Equivalent to the "cost-effectiveness" of improving data purity)
        best_attr = attributes[0]
        best_ratio = 0
        first = True

        for attr in attributes:
            gain = information_gain(data, attr)
            si = split_information(data, attr)

            if si == 0:
                ratio = 0
            else:
                ratio = gain / si

            if first or ratio > best_ratio:
                best_ratio = ratio
                best_attr = attr
                first = False

        tree = {
            "attribute": best_attr,
            "children": {},
            "majority": majority_class(data)
        }

        # get value of best_attr
        values = []
        for item in data:
            val = item[0][best_attr]
            if val not in values:
                values.append(val)

        for val in values:
            subset = []
            for item in data:
                if item[0][best_attr] == val:
                    subset.append(item)
            
            # remove used features
            new_attributes = []
            for a in attributes:
                if a != best_attr:
                    new_attributes.append(a)
            
            # dealing with children
            if len(subset) == 0:
                tree["children"][val] = majority_class(data)
            else:
                tree["children"][val] = build_tree(subset, new_attributes)

        return tree

    # classification for testing
    def classify(tree, example):
        if type(tree) == str:
            return tree

        attr = tree["attribute"]
        value = example[attr]

        if value in tree["children"]:
            return classify(tree["children"][value], example)
        else:
            return tree["majority"]

    # print tree
    def print_tree(tree, depth=0):
        if type(tree) == str:
            print("  " * depth + "-> " + tree)
            return

        attr = tree["attribute"]
        for val in tree["children"]:
            print("  " * depth + f"[attr {attr} = {val}]")
            print_tree(tree["children"][val], depth + 1)

    # get attributes and construct trees
    num_attributes = len(training_data[0][0])
    attributes = []
    for i in range(num_attributes):
        attributes.append(i)

    tree = build_tree(training_data, attributes)

    # print_tree(default not to print first)

    # processing testing_data
    results = []
    for example in testing_data:
        label = classify(tree, example)
        results.append(label)

    return results


def create_folds(filename):

    died = []
    survived = []

    # read data
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.endswith("died"):
                died.append(line)
            else:
                survived.append(line)

    # create 10 folds
    folds = []
    for i in range(10):
        folds.append([])

    # replace died
    index = 0
    for item in died:
        folds[index % 10].append(item)
        index += 1

    # replace survived
    index = 0
    for item in survived:
        folds[index % 10].append(item)
        index += 1

    # write to file
    with open("heart-folds.csv", "w") as f:
        for i in range(10):
            f.write(f"fold{i+1}\n")
            for item in folds[i]:
                f.write(item + "\n")
            f.write("\n")


def load_folds(filename):
    # multiple temporary reads form the folds
    folds = []
    current_fold = []

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()

            # identify the starting point
            if line.startswith("fold"):
                if current_fold:
                    folds.append(current_fold)
                    current_fold = []

            # ignore empty lines
            elif line == "":
                continue

            else:
                current_fold.append(line)

        # for the last fold
        if current_fold:
            folds.append(current_fold)

    return folds


def calculate_accuracy(true_labels, predictions):
    # The number of predicted answers 
    # that match reality divided by the total number of data
    correct = 0

    for i in range(len(true_labels)):
        if true_labels[i] == predictions[i]:
            correct += 1

    return correct / len(true_labels)


def cross_validate(folds, classifier_func, k=None):
    accuracies = []

    for i in range(10):
        testing_fold = folds[i]

        # create training set
        training_data = []
        for j in range(10):
            if j != i:
                training_data.extend(folds[j])

        # create a temporary file for classifier reading
        with open("train_tmp.csv", "w") as f:
            for line in training_data:
                f.write(line + "\n")

        with open("test_tmp.csv", "w") as f:
            for line in testing_fold:
                features = ",".join(line.split(",")[:-1])
                f.write(features + "\n")

        # answer
        true_labels = []
        for line in testing_fold:
            true_labels.append(line.split(",")[-1])

        # call the model
        if k is not None:
            predictions = classifier_func("train_tmp.csv", "test_tmp.csv", k)
        else:
            predictions = classifier_func("train_tmp.csv", "test_tmp.csv")

        # calculate the accuracy
        acc = calculate_accuracy(true_labels, predictions)
        accuracies.append(acc)

        print(f"Fold {i+1} accuracy: {acc:.4f}")

    avg_acc = sum(accuracies) / len(accuracies)

    print("Average accuracy:", avg_acc)

    return avg_acc


def main():

    folds = load_folds("heart-folds.csv")

    print("=== KNN ===")
    cross_validate(folds, classify_knn, k=5)

    print("\n=== KNN+ ===")
    cross_validate(folds, classify_knnplus, k=5)

    print("\n=== Decision Tree ===")
    cross_validate(folds, classify_dt)

    print("\n=== Decision Tree+ ===")
    cross_validate(folds, classify_dtplus)


# output
if __name__ == "__main__":
    create_folds("heart.csv")
    main()