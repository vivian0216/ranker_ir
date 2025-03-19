import pyterrier as pt

from datasets import load_dataset

class GetDataset():
    '''
    This class is used to load the dataset from the IRDS.
    '''
    def __init__(self, dataset_name, dataset_type):
        self.dataset_name = dataset_name
        self.dataset_type = dataset_type

    def load_dataset(self):
        '''
        Load the dataset from the datasets library. This is for the queries and qrels.
        '''
        return load_dataset(self.dataset_name, self.dataset_type, trust_remote_code=True)

    def get_dataset(self):
        ''' 
        Get the dataset from the pyterrier library. This is for the documents.
        '''
        return pt.get_dataset(self.dataset_name)

if __name__ == '__main__':
    getQueries = GetDataset('irds/cord19_trec-covid', 'queries')
    queries = getQueries.load_dataset()
    
    getQrels = GetDataset('irds/cord19_trec-covid', 'qrels')
    qrels = getQrels.load_dataset()
    
    getDocs = GetDataset('irds:cord19/trec-covid', 'docs')
    docs = getDocs.get_dataset()
    
    # Example: Display a few queries
    for record in queries:
        print(record)
        break  # Show only one for now
    # Example: Display a few qrels
    for record in qrels:
        print(record)
        break
    # Example: Display a few docs
    for doc in docs.get_corpus_iter():
        print(doc)
        break
