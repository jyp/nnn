import tensorflow as tf


# optimize is one of tf.train.GradientDescentOptimizer(0.05), etc.
def train (sess, model, optimizer, train_generator, valid_generator, epochs):
    (x,y,y_,accuracy,loss) = model
    train = optimizer.minimize(loss) # must come before the initializer (this line creates variables!)
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    for i in range(epochs):
        totalLoss = 0
        n = 0
        for (x_train,y_train) in enumerate(train_generator()):
            (_,lossAcc) = sess.run([train,loss], feed_dict={x:x_train, y:y_train})
            n+=1
            totalLoss += lossAcc
        print ("Training Loss = ", totalLoss / float(n))

        totalLoss = 0
        totalAccur = 0
        n = 0
        for (n,(x_train,y_train)) in enumerate(valid_generator()):
            lossAcc,accur = sess.run([loss,accuracy], feed_dict={x:x_train, y:y_train})
            totalLoss += lossAcc
            totalAccur += totalAccur
            n+=1
        print ("Validation Loss = ", totalLoss / float(n), " Accuracy = ", totalAccur / float(n))


def predict (sess, model, x_generator):
    (x,y,y_,accuracy,loss) = model
    sess.run(init)
    for (n,i) in enumerate(x_generator()):
        sess.run(y, feed_dict={x:x_generator})
