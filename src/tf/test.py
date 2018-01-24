'''
Created on 2018年1月19日

@author: user
'''
import tensorflow as tf

_CSV_COLUMN_DEFAULTS = [[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0]]
feature_names = ['f1','f2','f3','f4','f5','f6','f7','f8']
feature_columns= [tf.feature_column.numeric_column(k) for k in feature_names]

train_file_path = r'train.txt'
eval_file_path  = r'test.txt'
tf.logging.set_verbosity(tf.logging.INFO)
#输入函数
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
#模型函数
def cnn_fn(features,labels,mode):
    # Create the layer of input
    input_layer = tf.reshape(tf.feature_column.input_layer(features, feature_columns),[-1,1,8,1])
    
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=2,
        kernel_size=[1, 2],
        padding="same",
        activation=tf.nn.relu)
    
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[1, 2], strides=[2,1])
    
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=2,
        kernel_size=[1, 3],
        padding="same",
        activation=tf.nn.relu)
 
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[1, 2], strides=[2,1])
    pool2_flat = tf.reshape(pool2, [-1, 6*2])
    
    dense = tf.layers.dense(inputs=pool2_flat, units=128, activation=tf.tanh)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
    
    logits = tf.layers.dense(inputs=dropout, units=2)
    
    predictions = {
# Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
# Add `softmax_tensor` to the graph. It is used for PREDICT and by the
# `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

# Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

# Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

# Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    
    pass
def main():
    tf.logging.info("Before classifier construction")
    
    classifier = tf.estimator.Estimator(
        model_fn=cnn_fn)  # Path to where checkpoints etc are stored
    
    tf.logging.info("...done constructing classifier")
    tf.logging.info("Before classifier.train")
    
    classifier.train(
        input_fn=lambda:input_fn(train_file_path,True,10))
    
    tf.logging.info("...done classifier.train")
    tf.logging.info("Before classifier.evaluate")
    evaluate_result = classifier.evaluate(
        input_fn=lambda:input_fn(eval_file_path, False))
    tf.logging.info("...done classifier.evaluate")
    tf.logging.info("Evaluation results")
    
    for key in evaluate_result:
        tf.logging.info("   {}, was: {}".format(key, evaluate_result[key]))
        
    pass
if '__main__' == __name__:
    main()