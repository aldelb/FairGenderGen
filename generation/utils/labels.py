import torch
from collections import Counter
import random


# Tableau de labels
categories_gender = ["H", "F", "silence"]
categories_small_gender = ["H", "F"]


# Créer les dictionnaires de mapping
label_to_index_gender  = {label: i for i, label in enumerate(sorted(categories_gender))}
index_to_label_gender  = {i: label for label, i in label_to_index_gender.items()}

label_to_index_small_gender  = {label: i for i, label in enumerate(sorted(categories_small_gender))}
index_to_label_small_gender  = {i: label for label, i in label_to_index_small_gender.items()}

print("*"*10, "LABELS")
print(label_to_index_gender)
print(index_to_label_gender)

# Fonction pour convertir les labels en représentation one-hot
def label_to_one_hot(label, type):
    if type == "gender":
        label_to_index = label_to_index_gender
    elif type == "small_gender":
        label_to_index = label_to_index_small_gender
    num_classes = len(label_to_index)
    one_hot = torch.zeros(num_classes)
    one_hot[label_to_index[label]] = 1
    return one_hot

# Fonction pour récupérer le label à partir de la représentation one-hot
def one_hot_to_label(one_hot, type):
    if type == "gender":
        index_to_label = index_to_label_gender
    elif type == "small_gender":
        index_to_label = index_to_label_small_gender
    index = torch.argmax(one_hot)
    return index_to_label[index.item()]

def one_hot_to_index(one_hot, type):
    index = torch.argmax(one_hot)
    if type == "gender":
        index_to_label = index_to_label_gender
        return label_to_index_gender[index_to_label[index.item()]]
    elif type == "small_gender":
        index_to_label = index_to_label_small_gender
        return label_to_index_small_gender[index_to_label[index.item()]]

def other_label(categories, current_label_list, type):
    new_list = []
    for one_hot_current_label in current_label_list:
        current_label = one_hot_to_label(one_hot_current_label, type)
        list_wt_label = [l for l in categories if l != current_label]
        new_label = random.choice(list_wt_label)
        new_list.append(label_to_one_hot(new_label, type))
    return torch.stack(new_list)

def get_other_label(label_list, type):
    if type == "gender":
        return other_label(categories_gender, label_list, type)
    elif type == "small_gender":
        return other_label(categories_small_gender, label_list, type)


def get_labels(type):
    if type == "gender":
        labels = categories_gender
    elif type == "small_gender":
        labels = categories_small_gender
    return labels

def get_color(type):
    if type == "gender":
        color = {"silence": "grey", "H": "red", "F" : "green"}
    elif type == "small_gender":
        color = {"H": "red", "F" : "green"}
    return color

def get_labels_to_index(type):
    if type == "gender":
        labels = label_to_index_gender
    elif type == "small_gender":
        labels = label_to_index_small_gender
    return labels


def get_maj_label(labels):
        # Count the number of occurrences of each label
        label_counts = Counter(labels)
        # Find the majority label
        majority_label = max(label_counts, key=label_counts.get)
        # Calculate the percentage presence of the majority label
        percentage_majority = label_counts[majority_label] / len(labels) * 100
        # If there's something other than silence, we'll take something else.
        if majority_label == "silence" and percentage_majority < 100:
            majority_label = Counter(labels).most_common(2)[1][0]
            #second_percentage_majority = label_counts[second_majority_label] / len(labels) * 100
        return majority_label


def supress_silence_index(data, one_hot_labels_list, type):
    raw_labels_list = [one_hot_to_label(label, type) for label in one_hot_labels_list]
    supress_index = []
    for i in range(len(raw_labels_list)):
        if "silence" in raw_labels_list[i]:
            supress_index.append(i)
    tensor = data.clone()
    masque = torch.ones(data.size(0), dtype=torch.bool).to(tensor)
    masque[supress_index] = False
    tensor_without_silence = torch.index_select(tensor, dim=0, index=torch.nonzero(masque).squeeze()).to(tensor)
    one_hot_labels_without_silence = torch.index_select(one_hot_labels_list, dim=0, index=torch.nonzero(masque).squeeze()).to(tensor)
    labels_without_silence = [raw_labels_list[i] for i in range(len(raw_labels_list)) if i not in supress_index]
    print("******len after supress silence**********")
    print(len(tensor_without_silence), len(labels_without_silence))
    return tensor_without_silence, one_hot_labels_without_silence

def get_no_silence_index_from_one_hot(one_hot_labels_list, type):
    raw_labels_list = [one_hot_to_label(label, type) for label in one_hot_labels_list]
    index_no_silence = [index for index, element in enumerate(raw_labels_list) if element != "silence"]
    return index_no_silence
