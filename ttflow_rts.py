import tensorflow as tf


# optimize is one of tf.train.GradientDescentOptimizer(0.05), etc.
def train (sess, model, optimizer, train_generator, valid_generator, epochs):
    (x,y,y_,accuracy,loss) = model
    sess.run(tf.global_variables_initializer())
    train = optimizer.minimize(loss)
    for i in range(epochs):
        totalLoss = 0
        for (x_train,y_train) in train_generator():
            (_,loss) = sess.run([train,loss], feed_dict={x:x_train, y:y_train})
            totalLoss += loss
        print ("Training Loss = ", totalLoss / float(epochs))

        totalLoss = 0
        for (x_train,y_train) in valid_generator():
            loss,accur = sess.run([loss,accuracy], feed_dict={x:x_train, y:y_train})
        print ("Validation Loss = ", totalLoss / float(epochs), " Accuracy = ", accuracy / float(epochs))


def predict (sess, model, x_generator):
    (x,y,y_,accuracy,loss) = model
    sess.run(init)
    for i in range(epochs):
        sess.run(y, feed_dict={x:x_generator})
