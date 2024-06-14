import tensorflow as tf
from tensorflow.keras import backend as K


class CRAFTLossV1(tf.keras.losses.Loss):
    def __init__(self, 
                 reduction='auto', 
                 name='CRAFTLossV1'):
        super(CRAFTLossV1, self).__init__(reduction=reduction, name=name)
        self.invariant_name = "craft_loss_v1"
        self.coefficient = 1
    
    def mse(self, y_true, y_pred):
        return tf.square(y_true - y_pred)
        
    def __call__(self, y_true, y_pred, sample_weight=None):
        region_scores, affinity_scores, confidence_masks = y_true
        region_predict, affinity_predict = y_pred
        loss1 = self.mse(region_scores, region_predict)
        loss2 = self.mse(affinity_scores, affinity_predict)
        loss = char_loss + affi_loss
        return tf.reduce_mean(loss) * self.coefficient