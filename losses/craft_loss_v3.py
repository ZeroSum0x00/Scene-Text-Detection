import tensorflow as tf
from tensorflow.keras import backend as K


# class CRAFTLossV3(tf.keras.losses.Loss):
#     def __init__(self, 
#                  negative_ratio=1,
#                  num_min_negative=1000,
#                  reduction='auto', 
#                  name='CRAFTLossV3'):
#         super(CRAFTLossV3, self).__init__(reduction=reduction, name=name)
#         self.negative_ratio = negative_ratio
#         self.num_min_negative = num_min_negative
#         self.invariant_name = "craft_loss_v3"
#         self.coefficient = 1
    
#     def mse(self, y_true, y_pred):
#         return tf.square(y_true - y_pred)
    
#     def single_image_loss(self, y_true, y_pred):
#         batch_size = y_true.shape[0]
#         positive_loss, negative_loss = 0, 0
        
#         for single_label, single_loss in zip(y_true, y_pred):
#             # positive_loss
#             pos_pixel = tf.where(tf.greater_equal(single_label, 0.1), 1, 0)
#             pos_pixel = tf.cast(pos_pixel, dtype=tf.float32)
#             n_pos_pixel = tf.reduce_sum(pos_pixel)
#             pos_loss_region = single_loss * pos_pixel
#             positive_loss += tf.reduce_sum(pos_loss_region) / max(n_pos_pixel, 1e-12)

#             # negative_loss
#             neg_pixel = tf.where(tf.less(single_label, 0.1), 1, 0)
#             neg_pixel = tf.cast(neg_pixel, dtype=tf.float32)
#             n_neg_pixel = tf.reduce_sum(neg_pixel)            
#             neg_loss_region = single_loss * neg_pixel
            
#             if n_pos_pixel != 0:
#                 if n_neg_pixel < self.negative_ratio * n_pos_pixel:
#                     negative_loss += tf.reduce_sum(neg_loss_region) / n_neg_pixel
#                 else:
#                     n_hard_neg = max(self.num_min_negative, self.negative_ratio * n_pos_pixel)
#                     negative_loss += tf.reduce_sum(
#                                         tf.math.top_k(tf.reshape(neg_loss_region, [-1]), k=int(n_hard_neg))[0] / n_hard_neg
#                                      )                    
#             else:
#                 # only negative pixel
#                 negative_loss += tf.reduce_sum(
#                                     tf.math.top_k(tf.reshape(neg_loss_region, [-1]), k=int(self.num_min_negative))[0] / self.num_min_negative
#                                  )
#         total_loss = (positive_loss + negative_loss) / batch_size
#         return total_loss
        
#     def __call__(self, y_true, y_pred, sample_weight=None):
#         region_scores, affinity_scores, confidence_masks = y_true
#         region_predict, affinity_predict = y_pred
#         loss1 = self.mse(region_scores, region_predict)
#         loss2 = self.mse(affinity_scores, affinity_predict)
        
#         loss_region = tf.multiply(loss1, confidence_masks)
#         loss_affinity = tf.multiply(loss2, confidence_masks)
        
#         char_loss = self.single_image_loss(region_scores, loss_region)
#         affi_loss = self.single_image_loss(affinity_scores, loss_affinity)
#         loss = char_loss + affi_loss
#         return tf.reduce_mean(loss) * self.coefficient


class CRAFTLossV3(tf.keras.losses.Loss):
    def __init__(self, 
                 negative_ratio=1,
                 num_min_negative=1000,
                 reduction='auto', 
                 name='CRAFTLossV3'):
        super(CRAFTLossV3, self).__init__(reduction=reduction, name=name)
        self.negative_ratio = negative_ratio
        self.num_min_negative = num_min_negative
        self.invariant_name = "craft_loss_v3"
        self.coefficient = 1
    
    def mse(self, y_true, y_pred):
        return tf.square(y_true - y_pred)
    
    def single_image_loss(self, y_true, y_pred):
        batch_size = tf.shape(y_true)[0]
        positive_loss, negative_loss = 0, 0
        
        def positive_loss_calculator(i, label, predict, result):
            pos_pixel = tf.where(tf.greater_equal(label[i], 0.1), 1, 0)
            pos_pixel = tf.cast(pos_pixel, dtype=tf.float32)
            n_pos_pixel = tf.reduce_sum(pos_pixel)
            pos_loss_region = predict[i] * pos_pixel
            pos_loss_region = tf.cast(pos_loss_region, dtype=tf.float32)
            positive_loss = tf.reduce_sum(pos_loss_region) / tf.math.maximum(n_pos_pixel, 1e-12)
            result = result.write(i, positive_loss)
            return i+1, label, predict, result

        def negative_loss_calculator(i, label, predict, result):
            neg_pixel = tf.where(tf.less(label[i], 0.1), 1, 0)
            neg_pixel = tf.cast(neg_pixel, dtype=tf.float32)
            n_neg_pixel = tf.reduce_sum(neg_pixel)
            n_pos_pixel = tf.reduce_sum(tf.ones_like(label[i])) - n_neg_pixel
            neg_loss_region = predict[i] * neg_pixel
            
            n_hard_neg = tf.math.maximum(tf.constant(self.num_min_negative, dtype=tf.float32), self.negative_ratio * n_pos_pixel)
            true_fn2 = lambda: tf.reduce_sum(neg_loss_region) / n_neg_pixel
            false_fn2 = lambda: tf.reduce_sum(tf.math.top_k(tf.reshape(neg_loss_region, [-1]), k=tf.cast(n_hard_neg, dtype=tf.int32))[0] / n_hard_neg)
            true_fn1 = lambda: tf.cond(tf.less(n_neg_pixel, self.negative_ratio * n_pos_pixel), true_fn2, false_fn2)
            false_fn1 = lambda: tf.reduce_sum(tf.math.top_k(tf.reshape(neg_loss_region, [-1]), k=int(self.num_min_negative))[0] / self.num_min_negative)
            negative_loss = tf.cond(tf.not_equal(n_pos_pixel, 0), true_fn1, false_fn1)
            # if n_pos_pixel != 0:
            #     if n_neg_pixel < self.negative_ratio * n_pos_pixel:
            #         negative_loss = tf.reduce_sum(neg_loss_region) / n_neg_pixel
            #     else:
            #         n_hard_neg = tf.math.maximum(self.num_min_negative, self.negative_ratio * n_pos_pixel)
            #         negative_loss = tf.reduce_sum(
            #                            tf.math.top_k(tf.reshape(neg_loss_region, [-1]), k=int(n_hard_neg))[0] / n_hard_neg
            #                         )                    
            # else:
            #     # only negative pixel
            #     negative_loss = tf.reduce_sum(
            #                        tf.math.top_k(tf.reshape(neg_loss_region, [-1]), k=int(self.num_min_negative))[0] / self.num_min_negative
            #                     )
            result = result.write(i, negative_loss)
            return i+1, label, predict, result
        
        positive_loss_result = tf.TensorArray(dtype=tf.float32, size=batch_size, dynamic_size=True, infer_shape=False)
        negative_loss_result = tf.TensorArray(dtype=tf.float32, size=batch_size, dynamic_size=True, infer_shape=False)
        cond = lambda i, *args: i < batch_size
        _, _, _, positive_loss_result = tf.while_loop(cond, positive_loss_calculator, [0, y_true, y_pred, positive_loss_result])
        _, _, _, negative_loss_result = tf.while_loop(cond, negative_loss_calculator, [0, y_true, y_pred, negative_loss_result])
        positive_loss = tf.reduce_sum(positive_loss_result.stack())
        negative_loss = tf.reduce_sum(negative_loss_result.stack())
        total_loss = (positive_loss + negative_loss) / tf.cast(batch_size, dtype=tf.float32)
        return total_loss
        
    def __call__(self, y_true, y_pred, sample_weight=None):
        region_scores, affinity_scores, confidence_masks = y_true
        region_predict, affinity_predict = y_pred
        loss1 = self.mse(region_scores, region_predict)
        loss2 = self.mse(affinity_scores, affinity_predict)
        
        loss_region = tf.multiply(loss1, confidence_masks)
        loss_affinity = tf.multiply(loss2, confidence_masks)
        
        char_loss = self.single_image_loss(region_scores, loss_region)
        affi_loss = self.single_image_loss(affinity_scores, loss_affinity)
        loss = char_loss + affi_loss
        return tf.reduce_mean(loss) * self.coefficient