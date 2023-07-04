from transformers import pipeline
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import torch
from utils.load_datasets import load_MR, load_Semeval2017A
from training import get_metrics_report

# DATASET = 'MR'
# PRETRAINED_MODEL = 'siebert/sentiment-roberta-large-english'

DATASET = 'Semeval2017A'
# PRETRAINED_MODEL = 'cardiffnlp/twitter-roberta-base-sentiment'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS_MAPPING = {
    'siebert/sentiment-roberta-large-english': {
        'POSITIVE': 'positive',
        'NEGATIVE': 'negative',
    },
    'cardiffnlp/twitter-roberta-base-sentiment': {
        'LABEL_0': 'negative',
        'LABEL_1': 'neutral',
        'LABEL_2': 'positive',
    },
    'textattack/bert-base-uncased-imdb': {
        'LABEL_0': 'negative',
        'LABEL_1': 'positive',
    },
    'textattack/bert-base-uncased-yelp-polarity': {
        'LABEL_0': 'negative',
        'LABEL_1': 'positive',
    },
    'mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis': {
        'positive': 'positive',
        'neutral': 'neutral',
        'negative': 'negative',
    },
    'Seethal/sentiment_analysis_generic_dataset': {
        'LABEL_0': 'negative',
        'LABEL_1': 'neutral',
        'LABEL_2': 'positive',
    },
    'cardiffnlp/twitter-xlm-roberta-base-sentiment': {
        'Positive': 'positive',
        'Neutral': 'neutral',
        'Negative': 'negative',
    },
    'j-hartmann/sentiment-roberta-large-english-3-classes': {
        'positive': 'positive',
        'neutral': 'neutral',
        'negative': 'negative',
    }
}

if __name__ == '__main__':
    # load the raw data
    if DATASET == "Semeval2017A":
        X_train, y_train, X_test, y_test = load_Semeval2017A()
        pretrained_models = ['j-hartmann/sentiment-roberta-large-english-3-classes']
        #'Seethal/sentiment_analysis_generic_dataset'
        #'cardiffnlp/twitter-roberta-base-sentiment',
        #'mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis',
        #'cardiffnlp/twitter-xlm-roberta-base-sentiment'

    elif DATASET == "MR":
        X_train, y_train, X_test, y_test = load_MR()
        pretrained_models = ['siebert/sentiment-roberta-large-english',
                             'textattack/bert-base-uncased-imdb',
                             'textattack/bert-base-uncased-yelp-polarity']
    else:
        raise ValueError("Invalid dataset")

    for PRETRAINED_MODEL in pretrained_models:
        # encode labels
        le = LabelEncoder()
        le.fit(list(set(y_train)))
        y_train = le.transform(y_train)
        y_test = le.transform(y_test)
        n_classes = len(list(le.classes_))

        # define a proper pipeline
        sentiment_pipeline = pipeline("sentiment-analysis", model=PRETRAINED_MODEL).to(DEVICE)

        y_pred = []
        for x in tqdm(X_test):
            # TODO: Main-lab-Q6 - get the label using the defined pipeline
            result = sentiment_pipeline(x)[0]
            label = result['label']
            y_pred.append(LABELS_MAPPING[PRETRAINED_MODEL][label])

        y_pred = le.transform(y_pred)
        print(f'\nDataset: {DATASET}\nPre-Trained model: {PRETRAINED_MODEL}\nTest set evaluation\n{get_metrics_report([y_test], [y_pred])}')
