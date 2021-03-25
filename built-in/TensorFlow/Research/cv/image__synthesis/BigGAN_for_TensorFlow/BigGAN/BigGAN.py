import time
from ops import *
from utils import *
from tensorflow.contrib.data import prefetch_to_device, shuffle_and_repeat, map_and_batch

class BigGAN(object):

    def __init__(self, sess, args):
        self.model_name = "BigGAN"  # name for checkpoint
        self.sess = sess
        self.dataset_name = args.dataset
        self.checkpoint_dir = args.checkpoint_dir
        self.sample_dir = args.sample_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir

        self.epoch = args.epoch
        self.iteration = args.iteration
        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq
        self.img_size = args.img_size

        """ Generator """
        self.layer_num = int(np.log2(self.img_size)) - 3
        self.z_dim = args.z_dim  # dimension of noise-vector
        self.gan_type = args.gan_type

        """ Discriminator """
        self.n_critic = args.n_critic
        self.sn = args.sn
        self.ld = args.ld
        self.args =args


        self.sample_num = args.sample_num  # number of generated images to be saved
        self.test_num = args.test_num


        # train
        self.g_learning_rate = args.g_lr
        self.d_learning_rate = args.d_lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2

        self.custom_dataset = False

        if self.dataset_name == 'mnist' :
            self.c_dim = 1
            self.data = load_mnist(size=self.img_size)

        elif self.dataset_name == 'cifar10' :
            self.c_dim = 3
            self.data = load_cifar10(size=self.img_size)

        else:
            self.c_dim = 3
            print('------dataset ----', self.dataset_name)
            self.data = load_data(dataset_name=self.dataset_name, size=self.img_size)
            print("----self.data ---", self.data)
            self.custom_dataset = True

        if self.args.phase == 'test':
            self.custom_dataset = False


        self.dataset_num = len(self.data)

        self.sample_dir = os.path.join(self.sample_dir, self.model_dir)
        check_folder(self.sample_dir)

        print()

        print("##### Information #####")
        print("# gan type : ", self.gan_type)
        print("# dataset : ", self.dataset_name)
        print("# dataset number : ", self.dataset_num)
        print("# batch_size : ", self.batch_size)
        print("# epoch : ", self.epoch)
        print("# iteration per epoch : ", self.iteration)

        print()

        print("##### Generator #####")
        print("# generator layer : ", self.layer_num)

        print()

        print("##### Discriminator #####")
        print("# discriminator layer : ", self.layer_num)
        print("# the number of critic : ", self.n_critic)
        print("# spectral normalization : ", self.sn)

    ##################################################################################
    # Generator
    ##################################################################################

    def generator(self, z, is_training=True, reuse=False):
        with tf.variable_scope("generator", reuse=reuse):
            ch = 1024
            x = fully_connected(z, units=4 * 4 * ch, sn=self.sn, scope='fc')
            x = tf.reshape(x, [-1, 4, 4, ch])

            x = up_resblock(x, channels=ch, is_training=is_training, sn=self.sn, scope='front_resblock_0')

            for i in range(self.layer_num // 2) :
                x = up_resblock(x, channels=ch // 2, is_training=is_training, sn=self.sn, scope='middle_resblock_' + str(i))
                ch = ch // 2

            x = self.google_attention(x, channels=ch, scope='self_attention')

            for i in range(self.layer_num // 2, self.layer_num) :
                x = up_resblock(x, channels=ch // 2, is_training=is_training, sn=self.sn, scope='back_resblock_' + str(i))
                ch = ch // 2

            x = batch_norm(x, is_training)
            x = relu(x)

            x = conv(x, channels=self.c_dim, kernel=3, stride=1, pad=1, pad_type='zero', scope='g_logit')
            x = tanh(x)
            # x = tf.identity(x, name='fake_image')

            return x

    ##################################################################################
    # Discriminator
    ##################################################################################

    def discriminator(self, x, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse):
            ch = 64

            x = init_down_resblock(x, channels=ch, sn=self.sn, scope='init_resblock')

            x = down_resblock(x, channels=ch * 2, sn=self.sn, scope='front_down_resblock')
            x = self.google_attention(x, channels=ch * 2, scope='self_attention')

            ch = ch * 2

            for i in range(self.layer_num) :
                if i == self.layer_num - 1 :
                    x = down_resblock(x, channels=ch, sn=self.sn, to_down=False, scope='middle_down_resblock_' + str(i))
                else :
                    x = down_resblock(x, channels=ch * 2, sn=self.sn, scope='middle_down_resblock_' + str(i))

                ch = ch * 2

            x = lrelu(x, 0.2)

            x = global_sum_pooling(x)

            x = fully_connected(x, units=1, sn=self.sn, scope='d_logit')

            return x

    def attention(self, x, channels, scope='attention'):
        with tf.variable_scope(scope):
            f = conv(x, channels // 8, kernel=1, stride=1, sn=self.sn, scope='f_conv') # [bs, h, w, c']
            g = conv(x, channels // 8, kernel=1, stride=1, sn=self.sn, scope='g_conv') # [bs, h, w, c']
            h = conv(x, channels, kernel=1, stride=1, sn=self.sn, scope='h_conv') # [bs, h, w, c]

            # N = h * w
            s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True) # # [bs, N, N]

            beta = tf.nn.softmax(s)  # attention map

            o = tf.matmul(beta, hw_flatten(h)) # [bs, N, C]
            gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

            o = tf.reshape(o, shape=x.shape) # [bs, h, w, C]
            o = conv(o, channels, kernel=1, stride=1, sn=self.sn, scope='attn_conv')

            x = gamma * o + x

        return x

    def google_attention(self, x, channels, scope='attention'):
        with tf.variable_scope(scope):
            batch_size, height, width, num_channels = x.get_shape().as_list()
            f = conv(x, channels // 8, kernel=1, stride=1, sn=self.sn, scope='f_conv')  # [bs, h, w, c']
            f = max_pooling(f)

            g = conv(x, channels // 8, kernel=1, stride=1, sn=self.sn, scope='g_conv')  # [bs, h, w, c']

            h = conv(x, channels // 2, kernel=1, stride=1, sn=self.sn, scope='h_conv')  # [bs, h, w, c]
            h = max_pooling(h)

            # N = h * w
            s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

            beta = tf.nn.softmax(s)  # attention map

            o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
            gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

            o = tf.reshape(o, shape=[batch_size, height, width, num_channels // 2])  # [bs, h, w, C]
            o = conv(o, channels, kernel=1, stride=1, sn=self.sn, scope='attn_conv')
            x = gamma * o + x

        return x

    def gradient_penalty(self, real, fake):
        if self.gan_type == 'dragan' :
            shape = tf.shape(real)
            eps = tf.random_uniform(shape=shape, minval=0., maxval=1.)
            x_mean, x_var = tf.nn.moments(real, axes=[0, 1, 2, 3])
            x_std = tf.sqrt(x_var)  # magnitude of noise decides the size of local region
            noise = 0.5 * x_std * eps  # delta in paper

            # Author suggested U[0,1] in original paper, but he admitted it is bug in github
            # (https://github.com/kodalinaveen3/DRAGAN). It should be two-sided.

            alpha = tf.random_uniform(shape=[shape[0], 1, 1, 1], minval=-1., maxval=1.)
            interpolated = tf.clip_by_value(real + alpha * noise, -1., 1.)  # x_hat should be in the space of X

        else :
            alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
            interpolated = alpha*real + (1. - alpha)*fake

        logit = self.discriminator(interpolated, reuse=True)

        grad = tf.gradients(logit, interpolated)[0]  # gradient of D(interpolated)
        grad_norm = tf.norm(flatten(grad), axis=1)  # l2 norm

        GP = 0

        # WGAN - LP
        if self.gan_type == 'wgan-lp':
            GP = self.ld * tf.reduce_mean(tf.square(tf.maximum(0.0, grad_norm - 1.)))

        elif self.gan_type == 'wgan-gp' or self.gan_type == 'dragan':
            GP = self.ld * tf.reduce_mean(tf.square(grad_norm - 1.))

        return GP

    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        """ Graph Input """
        # images
        # tf.random.set_random_seed(1234)
        if self.custom_dataset:
            Image_Data_Class = ImageData(self.img_size, self.c_dim)
            print('--- self.data---', self.data)
            inputs = tf.data.Dataset.from_tensor_slices(self.data)
            inputs = inputs.repeat(self.dataset_num).apply(map_and_batch(Image_Data_Class.image_processing, self.batch_size, num_parallel_batches=16, drop_remainder=True))

            #inputs_iterator = inputs.make_one_shot_iterator()
            #inputs_iterator = inputs.make_initializable_iterator()

            #self.inputs = inputs_iterator.get_next()
            self.inputs = inputs
        else:
            self.inputs = tf.placeholder(tf.float32, [self.batch_size, self.img_size, self.img_size, self.c_dim], name='real_images')
        if self.args.phase == 'test':
            self.z = tf.placeholder(tf.float32, [1, 1, 1, self.z_dim], name='z')

            """ Loss Function """
            # output of D for real images
            # real_logits = self.discriminator(self.inputs)

            # output of D for fake images
            fake_images = self.generator(self.z)
            # fake_logits = self.discriminator(fake_images, reuse=True)
        else:
            rank_size = 8
            rank_id = int(os.getenv('DEVICE_ID'))
            print('train ranksize = %d, rankid = %d' % (rank_size, rank_id))
            self.inputs = self.inputs.shard(rank_size, rank_id)
            inputs_iterator = self.inputs.make_one_shot_iterator()

            self.inputs = inputs_iterator.get_next()
            # noises
            self.z = tf.placeholder(tf.float32, [self.batch_size, 1, 1, self.z_dim], name='z')

            """ Loss Function """
            # output of D for real images
            real_logits = self.discriminator(self.inputs)

            # output of D for fake images
            fake_images = self.generator(self.z)
            fake_logits = self.discriminator(fake_images, reuse=True)

            if self.gan_type.__contains__('wgan') or self.gan_type == 'dragan' :
                GP = self.gradient_penalty(real=self.inputs, fake=fake_images)
            else :
                GP = 0

            # get loss for discriminator
            self.d_loss = discriminator_loss(self.gan_type, real=real_logits, fake=fake_logits) + GP

            # get loss for generator
            self.g_loss = generator_loss(self.gan_type, fake=fake_logits)

            """ Training """
            # divide trainable variables into a group for D and a group for G
            t_vars = tf.trainable_variables()
            d_vars = [var for var in t_vars if 'discriminator' in var.name]
            g_vars = [var for var in t_vars if 'generator' in var.name]

            # optimizers
            self.d_optim = tf.train.AdamOptimizer(self.d_learning_rate, beta1=self.beta1, beta2=self.beta2).minimize(self.d_loss, var_list=d_vars)
            self.g_optim = tf.train.AdamOptimizer(self.g_learning_rate, beta1=self.beta1, beta2=self.beta2).minimize(self.g_loss, var_list=g_vars)

            """" Testing """
            # for test
            self.fake_images = self.generator(self.z, is_training=False, reuse=True)

            """ Summary """
            self.d_sum = tf.summary.scalar("d_loss", self.d_loss)
            self.g_sum = tf.summary.scalar("g_loss", self.g_loss)

    ##################################################################################
    # Train
    ##################################################################################

    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # graph inputs for visualize training results
        self.sample_z = np.random.uniform(-1, 1, size=(self.batch_size, 1, 1, self.z_dim))

        # saver to save model
        self.saver = tf.train.Saver(max_to_keep=100)

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.iteration)
            start_batch_id = checkpoint_counter - start_epoch * self.iteration
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        past_g_loss = -1.
        
        train_log_dir = './train.log'
        for epoch in range(start_epoch, self.epoch):
            # get batch data
            log_list = []
            for idx in range(start_batch_id, self.iteration):
                batch_z = np.random.uniform(-1, 1, [self.batch_size, 1, 1, self.z_dim])

                if self.custom_dataset :

                    train_feed_dict = {
                        self.z: batch_z
                    }

                else :
                    random_index = np.random.choice(self.dataset_num, size=self.batch_size, replace=False)
                    # batch_images = self.data[idx*self.batch_size : (idx+1)*self.batch_size]
                    batch_images = self.data[random_index]

                    train_feed_dict = {
                        self.inputs : batch_images,
                        self.z : batch_z
                    }

                # update D network
                _, summary_str, d_loss = self.sess.run([self.d_optim, self.d_sum, self.d_loss], feed_dict=train_feed_dict)
                self.writer.add_summary(summary_str, counter)

                # update G network
                g_loss = None
                if (counter - 1) % self.n_critic == 0:
                    _, summary_str, g_loss = self.sess.run([self.g_optim, self.g_sum, self.g_loss], feed_dict=train_feed_dict)
                    self.writer.add_summary(summary_str, counter)
                    past_g_loss = g_loss

                # display training status
                counter += 1
                if g_loss == None :
                    g_loss = past_g_loss
                if idx % 10 == 0:
                    print("Epoch: [%2d] [%5d/%5d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                        % (epoch, idx, self.iteration, time.time() - start_time, d_loss, g_loss))
                    log_list.append("Epoch={}, step={}, time={}, d_loss={:.8f}, g_loss={:.8f}\n".format(epoch, idx, time.time() - start_time, d_loss, g_loss))    

                # save training results for every 300 steps
                if np.mod(idx+1, self.print_freq) == 0:
                    samples = self.sess.run(self.fake_images, feed_dict={self.z: self.sample_z})
                    tot_num_samples = min(self.sample_num, self.batch_size)
                    manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                    manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                    save_images(samples[:manifold_h * manifold_w, :, :, :],
                                [manifold_h, manifold_w],
                                './' + self.sample_dir + '/' + self.model_name + '_train_{:02d}_{:05d}.png'.format(epoch, idx+1))

                if np.mod(idx+1, self.save_freq) == 0:
                    self.save(self.checkpoint_dir, counter)

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            self.save(self.checkpoint_dir, counter)

            # show temporal results
            # self.visualize_results(epoch)
            with open(train_log_dir,'w') as f:
                 f.writelines(log_list)

        # save model for final step
        self.save(self.checkpoint_dir, counter)

    @property
    def model_dir(self):
        return "{}_{}_{}_{}_{}_{}".format(
            self.model_name, self.dataset_name, self.gan_type, self.img_size, self.z_dim, self.sn)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def visualize_results(self, epoch):
        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        """ random condition, random noise """

        z_sample = np.random.uniform(-1, 1, size=(self.batch_size, 1, 1, self.z_dim))

        samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample})

        save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                    self.sample_dir + '/' + self.model_name + '_epoch%02d' % epoch + '_visualize.png')

    def test(self):
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        result_dir = os.path.join(self.result_dir, self.model_dir)
        check_folder(result_dir)

        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        """ random condition, random noise """

        for i in range(self.test_num) :
            z_sample = np.random.uniform(-1, 1, size=(self.batch_size, 1, 1, self.z_dim))

            samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample})
            
            for j in range(self.batch_size):
                test_save_images(samples[j, :, :, :],
                        [128,128],
                        result_dir + '/' + self.model_name + '_test_{}_{}.png'.format(i,j))
