import tensorflow as tf


class DiceLoss(tf.keras.losses.Loss):
    def __init__(self,
                 eps = 1e-6,
                 reduction=tf.losses.Reduction.AUTO,
                 name='DiceLoss'):
        super(DiceLoss, self).__init__(reduction=reduction, name=name)
        self.eps = eps
        self.invariant_name = "dice_loss"
        self.coefficient = 1

    def __call__(self, y_true, y_pred, sample_weight=None):
        label, mask = y_true
        pred = y_pred

        if tf.rank(pred) == 4:
            label = tf.squeeze(label, axis=-1)
            pred = tf.squeeze(pred, axis=-1)

        assert pred.shape == label.shape
        assert pred.shape == mask.shape

        if sample_weight is not None:
            assert sample_weight.shape == mask.shape
            mask = sample_weight * mask

        intersection = tf.reduce_sum(label * pred * mask)
        union = tf.reduce_sum(label * mask) + tf.reduce_sum(pred * mask) + self.eps
        loss = 1 - (2 * intersection / union) * self.coefficient
        assert loss <= 1
        return tf.reduce_mean(loss * self.coefficient)