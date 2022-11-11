import pynlpir
import jieba
import json
import re

ignore_chars = ["/","@","【","】","#",":","[","]"]

def loadStopWords(data_path):
    """
    功能：加载停用词
    """
    stop_words = []
    with open(data_path,"r") as fp:
        for line in fp.readlines():
            stop_words.append(line.strip())
    # print(stop_words)
    return stop_words

def weibo_process(content):
    """
    功能：清洗微博内容并分词
    """
    processed_content = []
    # Replaces URLs with the word [URL]
    content = re.sub(r'(https?|ftp|file|www\.)[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', '[URL]', content)
    # Replaces Email with the word [URL]
    content = re.sub(r'[a-zA-Z0-9_-]+@[a-zA-Z0-9_-]+[\.][a-zA-Z0-9_-]+', '[URL]', content)
    # Replaces user with the word FORWARD
    content = re.sub(r'(\/\/){0,1}@.*?(：|:| )', '[FORWARD]', content)
    # Replaces number  with the word [N]
    content = re.sub(r'\d+', '[N]', content)
    # Replace 2+ dots with space
    content = re.sub(r'[\.。…]{2,}', '。', content)
    # Replace 2+ ~~ 为 ~
    content = re.sub(r'~{2,}', '~', content)
    # Replace 2+ 叹号 为 一个叹号
    content = re.sub(r'[!！]{2,}', '!', content)
    # Replace 2+ 叹号 为 一个叹号
    content = re.sub(r'[？?]{2,}', '?', content)
    # 去掉 //
    content = re.sub(r'//', ' ', content)
    # 去掉 引号
    content = re.sub(r'["“”\'‘’]', '', content)

    pynlpir.open(encoding='utf_8', encoding_errors='ignore')
    segments = pynlpir.segment(content, pos_tagging=False)
    i = 1
    count = len(segments) - 1
    for segment in segments:
        if re.match(r'\s+', segment):  # 过滤掉空格
            i = i + 1
            continue
        segment = re.sub(r'@[\S]+', '[USER_MENTION]', segment)
        processed_content.append(segment.strip())
        if (i == count) & (segment == '[USER_MENTION]'):  # 过滤掉最后一个单独的字
            break
        i = i + 1
    pynlpir.close()
    return processed_content


def datasetProcess(org_path,save_path,stop_words):
    """
    功能：过滤出微博内容重点中文并进行分词
    """
    outcome = []
    with open(org_path,"r",encoding="utf-8") as fp:
        for idx,item in enumerate(json.load(fp)):
            print("processing item {}".format(idx))
            content = item.get("content")
            label = item.get("label")
            # content = "".join(regex.findall(chinese,content))
            seg_list = weibo_process(content)
            # seg_list = jieba.cut(content,cut_all=False)
            words = []
            for word in seg_list:
                if word in ignore_chars:
                    continue
                if word not in stop_words:
                    words.append(word)
            outcome.append({"content":words,"label":label})
    
    with open(save_path,"w",encoding="utf-8") as fp:
        json.dump(outcome,fp,ensure_ascii=False)

def getWordDict(data_path,min_count=5):
    """
    功能：构建单词词典
    """
    word2id = {}
    # 统计词频
    with open(data_path,"r",encoding="utf-8") as fp:
        for item in json.load(fp):
            for word in item['content']:
                if word2id.get(word) == None:
                    word2id[word] = 1
                else:
                    word2id[word] += 1
    # 过滤低频词
    vocab = set()
    for word,count in word2id.items():
        if count >= min_count:
            vocab.add(word)

    # 构成单词到索引的映射词典
    word2id = {"PAD":0,"UNK":1}
    length = 2
    for word in vocab:
        word2id[word] = length
        length += 1
    with open("datasets/word2id.json",'w',encoding="utf-8") as fp:
        json.dump(word2id,fp,ensure_ascii=False)

if __name__ == "__main__":
    # stop_words = loadStopWords(data_path="中文停用词词表.txt")
    # datasetProcess("datasets/train_org.txt","datasets/train.txt",stop_words)
    # datasetProcess("datasets/test_org.txt","datasets/test.txt",stop_words)
    getWordDict(data_path="datasets/train.txt",min_count=5)