import tensorflow as tf
import os
from ascendcv.layers import Conv2D, ActLayer, ConvModule, DCNPack, depth_to_space, resize, tf_split

from .base_model import VSR
from ascendvsr.layers.res_block import ResBlockNoBN


class EDVR(VSR):
    
    def __init__(self,
                 model_name,
                 scale,
                 num_frames,
                 data_dir,
                 set_file,
                 batch_size,
                 raw_size,
                 in_size,
                 output_dir,
                 solver,
                 is_train,
                 device,
                 is_distributed,
                 checkpoint,
                 cfg):
        super(EDVR, self).__init__(
            model_name,
            scale,
            num_frames,
            data_dir,
            set_file,
            batch_size,
            raw_size,
            in_size,
            output_dir,
            solver,
            is_train,
            device,
            is_distributed,
            checkpoint,
            cfg)
        edvr_cfg = cfg.edvr
        self.with_tsa = edvr_cfg.with_tsa
        self.mid_channels = edvr_cfg.mid_channels
        self.num_groups = edvr_cfg.num_groups
        self.num_deform_groups = edvr_cfg.num_deform_groups
        self.num_blocks_extraction = edvr_cfg.num_blocks_extraction
        self.num_blocks_reconstruction = edvr_cfg.num_blocks_reconstruction

    def feature_extraction(self, x, act_cfg=dict(type='LeakyRelu', alpha=0.1)):
        # extract LR features
        with tf.variable_scope('extraction', reuse=tf.AUTO_REUSE):
            # L1
            # l1_feat = tf.reshape(x, [-1, x.shape[2], x.shape[3], x.shape[4]])
            l1_feat = Conv2D(x, self.mid_channels, name='conv_first')
            l1_feat = ActLayer(act_cfg)(l1_feat)
            l1_feat = ResBlockNoBN(num_blocks=self.num_blocks_extraction, mid_channels=self.mid_channels)(l1_feat)
            # L2
            l2_feat = ConvModule(l1_feat, self.mid_channels, strides=[2, 2], act_cfg=act_cfg, name='feat_l2_conv1')
            l2_feat = ConvModule(l2_feat, self.mid_channels, act_cfg=act_cfg, name='feat_l2_conv2')
            # L3
            l3_feat = ConvModule(l2_feat, self.mid_channels, strides=[2, 2], act_cfg=act_cfg, name='feat_l3_conv1')
            l3_feat = ConvModule(l3_feat, self.mid_channels, act_cfg=act_cfg, name='feat_l3_conv2')

            # l1_feat_shape = l1_feat.get_shape().as_list()
            # l2_feat_shape = l2_feat.get_shape().as_list()
            # l3_feat_shape = l3_feat.get_shape().as_list()
            #
            # l1_feat = tf.reshape(l1_feat, [-1, self.num_frames, *l1_feat_shape[1:]])
            # l2_feat = tf.reshape(l2_feat, [-1, self.num_frames, *l2_feat_shape[1:]])
            # l3_feat = tf.reshape(l3_feat, [-1, self.num_frames, *l3_feat_shape[1:]])

            return l1_feat, l2_feat, l3_feat

    def pcd_align(self, neighbor_feats, ref_feats, act_cfg=dict(type='LeakyRelu', alpha=0.1)):

        with tf.variable_scope('pcd_align', reuse=tf.AUTO_REUSE):
            # The number of pyramid levels is 3.
            assert len(neighbor_feats) == 3 and len(ref_feats) == 3, (
                'The length of neighbor_feats and ref_feats must be both 3, '
                'but got {} and {}'.format(len(neighbor_feats), len(ref_feats)))

            # Pyramids
            upsampled_offset, upsampled_feat = None, None
            for i in range(3, 0, -1):
                with tf.variable_scope('level{}'.format(i)):
                    offset = tf.concat([neighbor_feats[i - 1], ref_feats[i - 1]], axis=-1)
                    offset = ConvModule(offset, self.mid_channels, act_cfg=act_cfg, name='offset_conv1')
                    if i == 3:
                        offset = ConvModule(offset, self.mid_channels, act_cfg=act_cfg, name='offset_conv2')
                    else:
                        offset = tf.concat([offset, upsampled_offset], axis=-1)
                        offset = ConvModule(offset, self.mid_channels, act_cfg=act_cfg, name='offset_conv2')
                        offset = ConvModule(offset, self.mid_channels, act_cfg=act_cfg, name='offset_conv3')

                    feat = DCNPack(neighbor_feats[i - 1], offset, self.mid_channels, kernel_size=[3, 3], padding='same',
                                   num_deform_groups=self.num_deform_groups, num_groups=self.num_groups,
                                   name='dcn_l{}'.format(i), dcn_version=self.cfg.edvr.dcn_version, impl=self.cfg.edvr.impl)
                    if i == 3:
                        feat = ActLayer(act_cfg)(feat)
                    else:
                        feat = tf.concat([feat, upsampled_feat], axis=-1)
                        feat = ConvModule(feat, self.mid_channels, act_cfg=act_cfg if i == 2 else None, name='feat_conv')

                    if i > 1:
                        # upsample offset and features
                        # upsampled_offset = tf.image.resize_bilinear(
                        upsampled_offset = resize(
                            offset, size=[offset.shape[1] * 2, offset.shape[2] * 2], align_corners=False,
                            name='upsample_offset{}'.format(i), method=self.cfg.edvr.upsampling)
                        upsampled_offset = upsampled_offset * 2
                        # upsampled_feat = tf.image.resize_bilinear(
                        upsampled_feat = resize(
                            feat, size=[feat.shape[1] * 2, feat.shape[2] * 2], align_corners=False,
                            name='upsample_feat{}'.format(i), method=self.cfg.edvr.upsampling)

            # Cascading
            offset = tf.concat([feat, ref_feats[0]], axis=-1)
            offset = ConvModule(offset, self.mid_channels, act_cfg=act_cfg, name='cas_offset_conv1')
            offset = ConvModule(offset, self.mid_channels, act_cfg=act_cfg, name='cas_offset_conv2')
            feat = DCNPack(feat, offset, self.mid_channels, kernel_size=[3, 3], padding='same',
                           num_deform_groups=self.num_deform_groups, name='dcn_cas',
                           dcn_version=self.cfg.edvr.dcn_version, impl=self.cfg.edvr.impl)
            feat = ActLayer(act_cfg)(feat)

            return feat

    def tsa_fusion(self, aligned_feat, act_cfg=dict(type='LeakyRelu', alpha=0.1)):
        with tf.variable_scope('tsa_fusion', reuse=tf.AUTO_REUSE):
            # temporal attention
            embedding_ref = Conv2D(aligned_feat[self.num_frames//2], self.mid_channels, name='temporal_attn1')

            corr_l = []  # correlation list
            for i in range(self.num_frames):
                emb = Conv2D(aligned_feat[i], self.mid_channels, name='temporal_attn2')
                emb = tf.cast(emb, tf.float32)
                corr = tf.reduce_sum(emb * embedding_ref, axis=-1, keep_dims=True)  # (n, h, w, 1)
                corr_l.append(corr)
            corr_prob = tf.nn.sigmoid(tf.stack(corr_l, axis=1))  # (n, t, h, w, 1)
            aligned_feat = tf.stack(aligned_feat, axis=1)
            aligned_feat = corr_prob * aligned_feat

            # fusion
            aligned_feat_shape = aligned_feat.get_shape().as_list()
            n, t, h, w, c = aligned_feat_shape
            aligned_feat = tf.transpose(aligned_feat, [0, 2, 3, 1, 4])
            aligned_feat = tf.reshape(aligned_feat, [-1, h, w, t*c])
            feat = ConvModule(aligned_feat, self.mid_channels, kernel_size=(1, 1), act_cfg=act_cfg, name='feat_fusion')

            # spatial attention
            attn = ConvModule(aligned_feat, self.mid_channels, kernel_size=(1, 1), act_cfg=act_cfg, name='spatial_attn1')
            attn_max = tf.nn.max_pool2d(attn, 3, 2, 'SAME')
            attn_avg = tf.nn.avg_pool(attn, 3, 2, 'SAME')
            attn = ConvModule(tf.concat([attn_max, attn_avg], axis=-1), self.mid_channels, kernel_size=(1, 1), act_cfg=act_cfg, name='spatial_attn2')
            # pyramid levels
            attn_level = ConvModule(attn, self.mid_channels, kernel_size=(1, 1), act_cfg=act_cfg, name='spatial_attn_l1')
            attn_max = tf.nn.max_pool2d(attn_level, 3, 2, 'SAME')
            attn_avg = tf.nn.avg_pool(attn_level, 3, 2, 'SAME')
            attn_level = ConvModule(tf.concat([attn_max, attn_avg], axis=-1), self.mid_channels, act_cfg=act_cfg, name='spatial_attn_l2')
            attn_level = ConvModule(attn_level, self.mid_channels, act_cfg=act_cfg, name='spatial_attn_l3')
            # attn_level = tf.image.resize_bilinear(
            attn_level = resize(
                attn_level, size=[attn_level.shape[1] * 2, attn_level.shape[2] * 2], align_corners=False,
                name='upsample1', method=self.cfg.edvr.upsampling)

            attn = ConvModule(attn, self.mid_channels, act_cfg=act_cfg, name='spatial_attn3') + attn_level
            attn = ConvModule(attn, self.mid_channels, kernel_size=(1, 1), act_cfg=act_cfg, name='spatial_attn4')
            # attn = tf.image.resize_bilinear(
            attn = resize(
                attn, size=[attn.shape[1] * 2, attn.shape[2] * 2], align_corners=False,
                name='upsample2', method=self.cfg.edvr.upsampling)
            attn = Conv2D(attn, self.mid_channels, name='spatial_attn5')
            attn = ConvModule(attn, self.mid_channels, kernel_size=(1, 1), act_cfg=act_cfg, name='spatial_attn_add1')
            attn_add = Conv2D(attn, self.mid_channels, kernel_size=(1, 1), name='spatial_attn_add2')
            
            attn = tf.cast(attn, tf.float32)
            attn = tf.nn.sigmoid(attn)
            
            feat = tf.cast(feat, tf.float32)
            attn_add = tf.cast(attn_add, tf.float32)

            # after initialization, * 2 makes (attn * 2) to be close to 1.
            feat = feat * attn * 2 + attn_add
            return feat

    def reconstruction(self, feat, x_center, act_cfg=dict(type='LeakyRelu', alpha=0.1)):
        # reconstruction
        with tf.variable_scope('reconstruction', reuse=tf.AUTO_REUSE):
            out = ResBlockNoBN(num_blocks=self.num_blocks_reconstruction, mid_channels=self.mid_channels)(feat)
            out = Conv2D(out, self.mid_channels * 2 ** 2, name='upsample1')
            out = depth_to_space(out, 2)
            out = Conv2D(out, self.mid_channels * 2 ** 2, name='upsample2')
            out = depth_to_space(out, 2)
            out = Conv2D(out, self.mid_channels, name='conv_hr')
            out = ActLayer(act_cfg)(out)
            out = Conv2D(out, 3, name='conv_last')
            
            base = resize(
                x_center, size=[x_center.shape[1] * 4, x_center.shape[2] * 4], align_corners=False,
                name='img_upsample', method=self.cfg.edvr.upsampling)
            base = tf.cast(base, tf.float32)
            out = tf.cast(out, tf.float32)
            out += base

            return out

    def build_generator(self, x):
        # shape of x: [B,T_in,H,W,C]
        with tf.variable_scope('G') as scope:
            # x_center = x[:, self.num_frames // 2]
            if self.cfg.model.input_format_dimension == 4:
                x_shape = x.get_shape().as_list()
                x = tf.reshape(x, [-1, self.num_frames, *x_shape[1:]])

            x_list = tf_split(x, self.num_frames, axis=1, keep_dims=False)
            x_center = x_list[self.num_frames//2]

            # extract LR features
            # l1_feat, l2_feat, l3_feat = self.feature_extraction(x)

            l1_feat_list = []
            l2_feat_list = []
            l3_feat_list = []
            for f in range(self.num_frames):
                l1_feat, l2_feat, l3_feat = self.feature_extraction(x_list[f])
                l1_feat_list.append(l1_feat)
                l2_feat_list.append(l2_feat)
                l3_feat_list.append(l3_feat)

            ref_feats = [  
                l1_feat_list[self.num_frames//2],
                l2_feat_list[self.num_frames//2],
                l3_feat_list[self.num_frames//2]
            ]
            aligned_feat = []
            for i in range(self.num_frames):
                neighbor_feats = [
                    l1_feat_list[i],
                    l2_feat_list[i],
                    l3_feat_list[i]
                ]
                aligned_feat.append(self.pcd_align(neighbor_feats, ref_feats))

            if self.with_tsa:
                feat = self.tsa_fusion(aligned_feat)
            else:
                aligned_feat = tf.stack(aligned_feat, axis=1)  # (n, t, h, w, c)
                aligned_feat_shape = aligned_feat.get_shape().as_list()
                last_dim = aligned_feat_shape[-1] * aligned_feat_shape[1]
                aligned_feat = tf.transpose(aligned_feat, [0, 2, 3, 1, 4])
                aligned_feat = tf.reshape(aligned_feat,
                                          [-1, aligned_feat.shape[1], aligned_feat.shape[2], last_dim])
                feat = Conv2D(aligned_feat, self.mid_channels, kernel_size=[1, 1], name='fusion')

            # reconstruction
            out = self.reconstruction(feat, x_center)

            return out

    def calculate_loss(self, SR, HR, **kwargs):
        eps = self.cfg.edvr.loss_margin
        reduction = self.cfg.edvr.loss_reduction
        
        SR = tf.cast(SR, tf.float32)
        HR = tf.cast(HR, tf.float32)

        if self.cfg.edvr.loss_type == 'marginal l1':
            losses = tf.maximum(tf.abs(SR - HR), eps)
        elif self.cfg.edvr.loss_type == 'l1':
            losses = tf.abs(SR - HR)
        elif self.cfg.edvr.loss_type == 'l2':
            losses = (SR - HR)**2
        elif self.cfg.edvr.loss_type == 'charbonnier':
            losses = tf.sqrt((SR - HR)**2 + eps)
        else:
            raise NotImplementedError

        losses = tf.reduce_sum(losses, axis=[1, 2, 3])
        if reduction == 'mean':
            return tf.reduce_mean(losses, axis=0)
        elif reduction == 'sum':
            return tf.reduce_sum(losses, axis=0)
        else:
            raise NotImplementedError

    # def freeze(self, sess_cfg):
    #     from tensorflow.python.framework import graph_util
    #     self.build()
    #
    #     with tf.Session(config=sess_cfg) as sess:
    #         print('[INFO] Loading trained model ...')
    #         self.load(sess)
    #         print('[INFO] Model loaded success.')
    #         print('[INFO] Freeze model to pb files')
    #
    #         pb_path = os.path.join(self.output_dir, '{}.pb'.format(type(self).__name__))
    #         try:
    #             constant_graph = graph_util.convert_variables_to_constants(
    #                 sess, sess.graph_def,
    #                 [self.SR.name.split(':')[0]]
    #             )
    #             with tf.gfile.FastGFile(pb_path, mode='wb') as f:
    #                 f.write(constant_graph.SerializeToString())
    #             print('[INFO] Model frozen success.')
    #         except Exception as e:
    #             print('[ERROR] Failed to freeze model.')
    #             print(e)
        