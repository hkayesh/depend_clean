from scripts.evaluation import Evaluator


if __name__ == '__main__':
    evaluator = Evaluator('files/srft_dataset.csv')

    # print segment level sentiment detection scores
    # evaluator.evaluate_sentiment_detection(scoring='f1_micro', merged=True)
    print(evaluator.get_category_counts(cat_type='sentiment', merged=True))








