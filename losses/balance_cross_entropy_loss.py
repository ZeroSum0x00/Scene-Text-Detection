import tensorflow as tf


class BalanceCrossEntropyLoss(tf.keras.losses.Loss):
    def __init__(self,
                 negative_ratio=3.0,
                 eps = 1e-6,
                 reduction=tf.losses.Reduction.AUTO,
                 name='BalanceCrossEntropyLoss'):
        super(BalanceCrossEntropyLoss, self).__init__(reduction=reduction, name=name)
        self.negative_ratio = negative_ratio
        self.eps = eps
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False, axis=0, reduction='none')
        self.invariant_name = "balance_cross_entropy_loss"
        self.coefficient = 1

    def __call__(self, y_true, y_pred, sample_weight=None):
        label, mask = y_true
        pred = y_pred

        if tf.rank(pred) == 4:
            label = tf.squeeze(label, axis=-1)
            pred = tf.squeeze(pred, axis=-1)

        positive = label * mask
        negative = (1 - label) * mask
        positive_count = tf.reduce_sum(positive)
        negative_count = tf.minimum(tf.reduce_sum(negative), positive_count * self.negative_ratio)
        loss = self.loss_fn(label, pred)
        positive_loss = loss * positive
        negative_loss = loss * negative

        negative_loss = tf.math.top_k(tf.reshape(negative_loss, [-1]), k=int(negative_count))[0]
        balance_loss  = (tf.reduce_sum(positive_loss) + tf.reduce_sum(negative_loss)) / (positive_count + negative_count + self.eps)
        return tf.reduce_mean(balance_loss * self.coefficient)