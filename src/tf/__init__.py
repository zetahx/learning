# -*- coding:utf-8 -*-
'''
Created on 2017年12月11日

@author: user
'''
import tensorflow as tf
import numpy as np
import time 
_CSV_COLUMN_DEFAULTS = [[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0]]
feature_names = ['f1','f2','f3','f4','f5','f6','f7','f8']

train_file_path = r'train.txt'
eval_file_path  = r'test.txt'

def input_fn(file_path,perform_shuffle=False, repeat_count=1):
    def parseCSVLine(line):
            parsed_line = tf.decode_csv(line, _CSV_COLUMN_DEFAULTS)
            label = parsed_line[-1]
            del parsed_line[-1]
            features = parsed_line
            d = dict(zip(feature_names, features))
            return d,label
        
    dataset = tf.data.TextLineDataset(file_path)
    dataset = dataset.map(lambda x:parseCSVLine(x))
    
    if perform_shuffle:
        # Randomizes input using a window of 256 elements (read into memory)
        dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.repeat(repeat_count) # Repeats dataset this # times
    dataset = dataset.batch(32)  # Batch size to use
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels
   

#dnn_estimator
estimator = tf.estimator.DNNClassifier(
    feature_columns= [tf.feature_column.numeric_column(k) for k in feature_names],
    hidden_units=[5,6],
    n_classes=2,
    optimizer=tf.train.AdamOptimizer()
#     dropout = 0.2
    )
print("start")
strat = time.time()

estimator.train(input_fn=lambda:input_fn(train_file_path,True,10))
metrics = estimator.evaluate(input_fn=lambda:input_fn(eval_file_path, False, 4))

end = time.time()

print("Evaluation results")
for key in metrics:
    print("   {}, was: {}".format(key, metrics[key]))
    
#计算用时
print ('use:')
print(end-strat)

print ('end')