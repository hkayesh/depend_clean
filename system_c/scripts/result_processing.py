import numpy as np
import matplotlib.pyplot as plt
from utilities import Utilities


class ResultProcessor:

    def __init__(self, plot=False):
        self.plot = plot
        self.utilities = Utilities()

    def get_total_positives_negatives(self, file_path, plot=False):
        rows = self.utilities.read_from_csv(file_path)

        all_sentiments = []
        for row in rows:
            del row[0]
            for item in row:
                all_sentiments.append(item.rsplit(' ', 1)[1])

        # for review in results.keys():
        #     all_sentiments = all_sentiments + results[review][2]

        positives = all_sentiments.count('positive')
        negatives = all_sentiments.count('negative')

        print("Total positive aspects: %d" % positives)
        print("Total negative aspects: %d" % negatives)

        # Pie chart, where the slices will be ordered and plotted counter-clockwise:
        labels = 'Positive', 'Negative'
        sizes = [positives, negatives]
        explode = (0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title('Total positive and negative feebacks (positives: %d, negatives: %d)' % (positives, negatives))

        if plot is True:
            plt.show()

    def per_class_sentiments(self, file_path, plot=False):
        rows = self.utilities.read_from_csv(file_path)

        per_class_data = {}
        for row in rows:
            del row[0]

            for item in row:
                sentiment = item.rsplit(' ', 1)[1]
                aspect = item.rsplit(' ', 1)[0].rsplit(' ', 1)[0]
                if aspect in per_class_data.keys():
                    per_class_data[aspect].append(sentiment)
                else:
                    per_class_data[aspect] = [sentiment]

        aspects = per_class_data.keys()
        positives = []
        negatives = []
        for aspect in aspects:
            positives.append(per_class_data[aspect].count('positive'))
            negatives.append(per_class_data[aspect].count('negative'))


        for aspect, positive, negative in zip(aspects, positives, negatives):
            print("%s : %d positive, %d negative" % (aspect, positive, negative))


        # plot as a barchart
        N = len(aspects)
        highest = max([x + y for x, y in zip(positives, negatives)])
        step = 5000
        ind = np.arange(N)  # the x locations for the groups
        width = 0.35  # the width of the bars: can also be len(x) sequence

        fig1, ax1 = plt.subplots()
        fig1.autofmt_xdate()
        p1 = plt.bar(ind, negatives, width, color='#d62728')
        p2 = plt.bar(ind, positives, width, bottom=negatives)

        plt.ylabel('Number of lebels')
        plt.xlabel('Aspects')
        plt.title('Sentiments by aspect categories')
        plt.xticks(ind, aspects)
        plt.yticks(np.arange(0, highest + step, step))
        plt.legend((p2[0], p1[0]), ('Positive', 'Negative'))

        # show pie chart for positive aspects per category
        positives_percentage = [0] * len(positives)
        if sum(positives) > 0:
            positives_percentage = [(float(n_positive)/sum(positives))*100 for n_positive in positives]

        # Pie chart, where the slices will be ordered and plotted counter-clockwise:
        labels = aspects
        sizes = positives_percentage
        explode = [0] * len(aspects)
        explode[positives_percentage.index(max(positives_percentage))] = 0.1

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title('Positive aspects per category')

        # show pie chart for negative aspects per category
        negatives_percentage = [0] * len(negatives)
        if sum(negatives) > 0:
            negatives_percentage = [(float(n_negative) / sum(negatives)) * 100 for n_negative in negatives]

        # Pie chart, where the slices will be ordered and plotted counter-clockwise:
        labels = aspects
        sizes = negatives_percentage
        explode = [0]*len(aspects)
        explode[negatives_percentage.index(max(negatives_percentage))] = 0.1

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title('Negative aspects per category')

        if plot is True:
            plt.show()
