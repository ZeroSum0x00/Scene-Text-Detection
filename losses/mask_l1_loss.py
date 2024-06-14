import tensorflow as tf


class MaskL1Loss(tf.keras.losses.Loss):
    def __init__(self,
                 reduction=tf.losses.Reduction.AUTO,
                 name='MaskL1Loss'):
        super(MaskL1Loss, self).__init__(reduction=reduction, name=name)
        self.invariant_name = "mask_l1_loss"
        self.coefficient = 1

    def __call__(self, y_true, y_pred, sample_weight=None):
        label, mask = y_true
        pred = y_pred
        mask_sum = tf.reduce_sum(mask)

        if mask_sum == 0:
            return mask_sum

        else:
            if tf.rank(pred) != tf.rank(label):
                pred = tf.squeeze(pred, axis=-1)

            loss = tf.math.abs(pred - label) * mask
            loss = tf.reduce_sum(loss) / mask_sum
            return tf.reduce_mean(loss * self.coefficient)