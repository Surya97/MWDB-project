from math import sqrt
from math import exp
from math import pi


def dividing_by_class(dataset):
	divided_dictionary = dict()
	for i in range(len(dataset)):
		feature = dataset[i]
		class_label = feature[-1]
		if (class_label not in divided_dictionary):
			divided_dictionary[class_label] = list()
		divided_dictionary[class_label].append(feature)
	return divided_dictionary


def mean(input):
	return sum(input) / float(len(input))


def stdev(input):
	avg = mean(input)
	variance = sum([(x - avg) ** 2 for x in input]) / float(len(input) - 1)
	return sqrt(variance)


def summarize_dataset(dataset):
	summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
	del (summaries[-1])
	return summaries


def division_by_class(dataset):
	separated = dividing_by_class(dataset)
	summaries = dict()
	for class_value, rows in separated.items():
		summaries[class_value] = summarize_dataset(rows)
	return summaries


def probability_calculation(x, mean, stdev):
	exponent = exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
	return (1 / (sqrt(2 * pi) * stdev)) * exponent


def class_label_probablities(summaries, row):
	total_rows = sum([summaries[label][0][2] for label in summaries])
	probabilities = dict()
	for class_value, class_summaries in summaries.items():
		probabilities[class_value] = summaries[class_value][0][2] / float(total_rows)
		for i in range(len(class_summaries)):
			mean, stdev, _ = class_summaries[i]
			probabilities[class_value] *= probability_calculation(row[i], mean, stdev)
	return probabilities

def predict(details, row):
	probabilities = class_label_probablities(details, row)
	best_label, best_prob = None, -1
	for class_value, probability in probabilities.items():
		if best_label is None or probability > best_prob:
			best_prob = probability
			best_label = class_value

	return best_label

