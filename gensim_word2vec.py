import json
import numpy as np
from gensim.models import word2vec

def load_pretrained_embedding(embedding_path):
    """
    功能：加载预训练词向量
    file_path：词嵌入向量路径
    """
    embedding_dict = {}
    with open(embedding_path,'r',encoding='utf-8') as fp:
        for line in fp.readlines():
            values = line.strip().split(' ')
            if len(values) <= 2:continue
            word = values[0]
            embedding_vec = np.asarray(values[1:],dtype=np.float32)
            embedding_dict[word] = embedding_vec
    
    return embedding_dict

def build_embdding_matrix(word_dict,embedding_path,embedding_dim):
    """
    加载词向量矩阵
    word_dict：根据数据集构造的单词词典
    embedding_dim：词嵌入向量的维度
    """
    embedding_dict= load_pretrained_embedding(embedding_path)
    vocab_size = len(word_dict)
    embedding_matrix = np.zeros((vocab_size,embedding_dim))
    for word,i in word_dict.items():
        embedding_vec = embedding_dict.get(word)
        if embedding_vec is not None:
            embedding_matrix[i] = embedding_vec
        elif word != "PAD":
            embedding_matrix[i] = np.random.randn(embedding_dim)
    return embedding_matrix

if __name__ == "__main__":
    train_path = "datasets/train.txt"
    sents = []
    with open(train_path,"r",encoding="utf-8") as fp:
        for item in json.load(fp):
            sents.append(item['content'])
    model = word2vec.Word2Vec(sents, vector_size=100, window=10, min_count=5,epochs=15,sg=1) 
    model.wv.save_word2vec_format('word2vec.bin',binary=False)



        
    