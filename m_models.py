import tensorflow as tf
from m_layers import *
from metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.test = None
        self.alphas = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for i in range(len(self.layers)):
            print("Processing GCN-{}-{}th layer".format(self.name, i))
            hidden = self.layers[i](self.activations[-1])
            if i == 3:
                # self.test = self.layers[i].test
                self.test = hidden
            self.activations.append(hidden)
        self.outputs = self.activations[-1]
        self._loss()

    def _loss(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "./output/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "./output/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class GCN(Model):
    def __init__(self, placeholders, input_dim, tag, length, parentvars, **kwargs):
        """
        Parameters
        ----------------
        tag: "user" or "item"
        """
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features_'+tag]
        self.input_dim = input_dim
        self.output_dim = FLAGS.output_dim
        self.placeholders = placeholders
        self.tag = tag
        self.length = length

        # todo weaving
        self.parentvars = parentvars

        self.build()

    def _loss(self):
        # Weight decay loss
        for i in range(len(self.layers)):
            for var in self.layers[i].vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

    def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            # output_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            length=self.length,
                                            placeholders=self.placeholders,
                                            tag=self.tag,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=False,
                                            logging=self.logging,
                                            name='first' + self.tag,
                                            featureless=True))

        # self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
        #                                     # output_dim=FLAGS.hidden1,
        #                                     output_dim=self.output_dim,
        #                                     length=self.length,
        #                                     placeholders=self.placeholders,
        #                                     tag=self.tag,
        #                                     act=tf.nn.relu,
        #                                     dropout=True,
        #                                     sparse_inputs=False,
        #                                     logging=self.logging,
        #                                     # name='first'+self.tag,
        #                                     featureless=False))
        #
        # self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
        #                                     output_dim=FLAGS.hidden2,
        #                                     length=self.length,
        #                                     placeholders=self.placeholders,
        #                                     tag=self.tag,
        #                                     act=tf.nn.relu,
        #                                     dropout=True,
        #                                     logging=self.logging))
        #
        # self.layers.append(GraphConvolution(input_dim=FLAGS.hidden2,
        #                                     output_dim=self.output_dim,
        #                                     length=self.length,
        #                                     placeholders=self.placeholders,
        #                                     tag=self.tag,
        #                                     act=tf.nn.relu,
        #                                     dropout=True,
        #                                     logging=self.logging))

        self.layers.append(SimpleAttLayer(attention_size=32,
                                          tag=self.tag,
                                          parentvars=self.parentvars,
                                          time_major=False))


class MOOCUM():
    def __init__(self, placeholders, input_dim_user, input_dim_item, user_dim, item_dim, learning_rate):
        """
        Parameters
        -----------------
        input_dim_user: user feature dim
        input_dim_item: item feature dim
        user dim: size of users
        item dim: size of items
        """
        self.name = "MOOCUM"
        self.placeholders = placeholders
        self.negative = placeholders['negative']
        self.length = user_dim
        self.user_dim = user_dim
        self.item_dim = item_dim

        self.vars = {}
        self.userModel = GCN(placeholders=self.placeholders, input_dim=input_dim_user, tag='user', length=user_dim,
                             parentvars=self.vars)
        self.itemModel = GCN(placeholders=self.placeholders, input_dim=input_dim_item, tag='item', length=item_dim,
                             parentvars=self.vars)
        self.user = self.userModel.outputs
        self.item = self.itemModel.outputs
        self.layers = []
        self.rate_matrix = None
        self.xuij = None
        self.result = None
        self.l2_loss = 0
        self.los = 0
        self.hrat1 = 0
        self.hrat5 = 0
        self.hrat10 = 0
        self.hrat20 = 0
        self.ndcg5 = 0
        self.ndcg10 = 0
        self.ndcg20 = 0
        self.mrr = 0
        self.err = None
        self.auc = 0
        # self.mse = 0
        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_op = None

        self.build()

    def build(self):
        self.layers.append(RateLayer(self.placeholders,
                                     self.user, self.item,
                                     user_dim=self.user_dim,
                                     item_dim=self.item_dim,
                                     parentvars=self.vars
                                     ))
        output = None
        for i in range(len(self.layers)):
            print("Using {} layer{}".format(self.name, i))
            output = self.layers[i]()
        self.rate_matrix, self.xuij, self.rate1, self.rate2, self.bias = output
        self.loss()
        self.train()
        self.env()

    def train(self):
        self.train_op = self.optimizer.minimize(self.los)

    def env(self):
        self.result = tf.nn.top_k(self.rate_matrix, 10).indices
        self.hrat()
        self.ndcg()
        self.mr()
        self.au()
        # self.ms()

    def loss(self):
        rating_matrix = self.placeholders['rating']
        # regularization in the paper
        self.l2_loss += self.userModel.loss # l2 loss from Using
        self.l2_loss += self.itemModel.loss # l2 loss from itemModel
        for i in range(len(self.layers)):
            for var in self.layers[i].vars.values():
                self.l2_loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        # add self vars
        for var in self.vars.values():
            self.l2_loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Debug Inf, NAN values  # Clip to get rid of Inf, NAN
        sigmoid_val = tf.sigmoid(self.xuij)
        # print("sigmoid val max", tf.reduce_max(sigmoid_val), "min", tf.reduce_min(sigmoid_val))
        self.los = -tf.reduce_mean(tf.log(tf.clip_by_value(sigmoid_val, 1e-10, 1.0))) + self.l2_loss

    def hrat(self):
        self.hrat1 = hr(self.rate_matrix, self.negative, self.length, k=1)
        self.hrat5 = hr(self.rate_matrix, self.negative, self.length, k=5)
        self.hrat10 = hr(self.rate_matrix, self.negative, self.length, k=10)
        self.hrat20 = hr(self.rate_matrix, self.negative, self.length, k=20)

    def ndcg(self):
        self.ndcg5 = ndcg(self.rate_matrix, self.negative, self.length, k=5)
        self.ndcg10 = ndcg(self.rate_matrix, self.negative, self.length, k=10)
        self.ndcg20 = ndcg(self.rate_matrix, self.negative, self.length, k=20)

    def mr(self):
        self.mrr = mrr(self.rate_matrix, self.negative, self.length)

    def au(self):
        self.auc = auc(self.rate_matrix, self.negative, self.length)

    def save(self, sess=None, info=""):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver()
        save_path = saver.save(sess, "./output/{}-{}.ckpt".format(self.name, info))
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        # saver = tf.train.Saver(self.vars)
        saver = tf.train.Saver()
        save_path = "./output/Mv.3/%s-besthr5.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)