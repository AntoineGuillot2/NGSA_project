import nltk
from nltk.corpus import reuters, stopwords


def load_doc_ids():
    try:
        documents = reuters.fileids()
    except LookupError:
        nltk.download('reuters')
        documents = reuters.fileids()

    doc_test = [d for d in documents if d.startswith('test/')]
    doc_train = [d for d in documents if d.startswith('training/')]

    return doc_train, doc_test


def load_doc_from_id(doc_id):
    doc = reuters.raw(doc_id)
    label = reuters.categories(doc_id)
    return doc, label