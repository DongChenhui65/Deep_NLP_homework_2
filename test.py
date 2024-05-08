import os
import random
import jieba
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score
from gensim import corpora, models
from collections import defaultdict
from sklearn.svm import SVC  # 以SVM作为分类器示例
import numpy
import re


def word_split(para):
    temp = []
    for x in para:
        temp.append(x)
    return temp


def delet_stopwords(word_list):
    stopwords_file_path = './/cn_stopwords.txt'
    stopword_file = open(stopwords_file_path, "r", encoding='utf-8')
    stop_words = stopword_file.read().split('\n')
    stopword_file.close()

    for word in stop_words:
        word_list = list(filter(lambda x: x != word, word_list))  # 删除所有的
    return word_list


# 处理段落和标签
def extract_paragraphs_and_labels(corpus_dict, num_paragraphs, k_value):
    result = []
    total_patragraphs_num = sum(len(corpus_dict[novel]) for novel in corpus_dict)
    for i in range(num_paragraphs):
        # 随机选择一个小说名字
        novel = random.choices(list(corpus_dict.keys()),
                               weights=[len(corpus_dict[key]) / total_patragraphs_num for key in corpus_dict])
        # 从该小说中选择一个段落
        paragraphs = corpus_dict[novel[0]]
        token_num = 0
        token = []
        while token_num < k_value:
            paragraph = random.choice(paragraphs)
            #temp = list(jieba.cut(paragraph))
            temp = word_split(paragraph)
            temp = delet_stopwords(temp)
            token.extend(temp)
            token_num = token_num + len(temp)
        token = token[:k_value]
        result.append((token, novel, k_value))
    return result


# 定义LDA算法
def latent_dirichlet_allocation(processed_data, topic_num):
    x = [item[0] for item in processed_data]  # 段落文本列表
    y = [item[1] for item in processed_data]  # 段落所属小说标签列表

    # 用train_test_split()函数进行测试集和训练集的划分
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    dictionary = corpora.Dictionary(x)  # 给每个词编号
    lda_corpus_train = [dictionary.doc2bow(tmp_doc) for tmp_doc in x_train]  # 返回一个元组列表，其中每个元组包含单词的ID和出现次数

    # 训练LDA模型
    lda = models.LdaModel(corpus=lda_corpus_train, id2word=dictionary, num_topics=topic_num)
    train_topic_distribution = lda.get_document_topics(lda_corpus_train)
    x_train_lda = np.zeros((len(x_train), topic_num))
    for i in range(len(train_topic_distribution)):
        tmp_topic_distribution = train_topic_distribution[i]
        for j in range(len(tmp_topic_distribution)):
            x_train_lda[i][tmp_topic_distribution[j][0]] = tmp_topic_distribution[j][1]

    classifier = SVC(kernel='linear', C=1, random_state=42)  # 使用SVC支持向量作为分类器
    classifier.fit(x_train_lda, y_train)

    # 用测试集进行试验
    lda_corpus_test = [dictionary.doc2bow(tmp_doc) for tmp_doc in x_test]
    test_topic_distribution = lda.get_document_topics(lda_corpus_test)
    x_test_lda = np.zeros((len(x_test), topic_num))
    for i in range(len(test_topic_distribution)):
        tmp_topic_distribution = test_topic_distribution[i]
        for j in range(len(tmp_topic_distribution)):
            x_test_lda[i][tmp_topic_distribution[j][0]] = tmp_topic_distribution[j][1]

    y_pred = classifier.predict(x_test_lda)
    y_pred = y_pred.reshape(-1, 1)
    # print("Classification Report:")
    # print(classification_report(y_test, y_pred))
    print("topic_num:", topic_num)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    # print("F1 Score (Macro):", f1_score(y_test, y_pred, average='macro'))


if __name__ == '__main__':
    corpus_dict = {}  # 假设这是您的语料库字典
    data_path = "./jyxstxtqj_downcc.com"
    file_list = os.listdir(data_path)
    # file_list = file_list[1:6]
    # 语料库预处理
    for file_name in file_list:
        file_path = data_path + '/' + file_name
        file = open(file_path, 'r', encoding="gb18030")
        merged_content = file.read()
        merged_content = merged_content.split("\n\u3000\u3000")
        corpus_dict[file_name.split('.')[0]] = merged_content
        file.close()

    num_paragraphs = 1000
    topic_num = [5, 10, 20, 30, 40, 50, 70, 100, 150, 200, 300, 1000, 3000]
    k_values = [10, 100, 500, 1000, 3000]
    processed_data = extract_paragraphs_and_labels(corpus_dict, num_paragraphs, k_values[0])
    latent_dirichlet_allocation(processed_data, topic_num[5])
    # for j in range(len(k_values)):
    #     k_value = k_values[j]
    #     processed_data = extract_paragraphs_and_labels(corpus_dict, num_paragraphs, k_value)
    #     print("----------------k_value:", k_value, "------------------------")
    #     for i in range(len(topic_num)):
    #         latent_dirichlet_allocation(processed_data, topic_num[i])
