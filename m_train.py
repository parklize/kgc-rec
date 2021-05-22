from m_utils import *
import tensorflow as tf
print(tf.__version__)
from m_models import *
import time
import numpy as np
from scipy import sparse

# ------------------------------------------
# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# ------------------------------------------
# Set params
learning_rate = .01
decay_rate = 1
global_steps = 500
decay_steps = 100
samples = 1024
batches = 30 # 856067/1024=837
print("learning rate:", learning_rate)
print("global steps:", global_steps)
print("samples:", samples)
print("batches", batches)

# ------------------------------------------
# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('hidden1', 256, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 128, 'Number of units in hidden layer 2.')
flags.DEFINE_float('dropout', 0, 'Dropout rate (1 - keep probability).') # default .5
flags.DEFINE_float('weight_decay', 1e-8, 'Weight for L2 loss on embedding matrix.') # default 5e-4
flags.DEFINE_integer('output_dim', 100, 'Output_dim of user final embedding.') # default 64, in paper it seems 100
flags.DEFINE_integer('latent_dim', 30, 'Latent_dim of user&item.')

# ------------------------------------------
# Load data
support_string_user = ['ucu', 'uvu', 'uctcu', 'uku']
support_string_item = ['kuk', 'kck']
rating, adjacency_matrix, features_item, features_user, support_user, support_item, negative = \
    load_data(user=support_string_user, item=support_string_item)

# User size item size
user_dim = rating.shape[0]
item_dim = rating.shape[1]

# Get non-zero indicies
straining_matrix = sparse.csr_matrix(rating)
uids, iids = straining_matrix.nonzero()
print("uids size", len(uids))

# user_support
support_num_user = len(support_string_user)
# item_support
support_num_item = len(support_string_item)
# Define placeholders
placeholders = {
    'rating': tf.placeholder(dtype=tf.float32, shape=rating.shape, name="rating"),
    'features_user': tf.placeholder(dtype=tf.float32, shape=features_user.shape, name='features_user'),
    'features_item': tf.placeholder(dtype=tf.float32, shape=features_item.shape, name="features_item"),
    'support_user': [tf.placeholder(dtype=tf.float32, name='support'+str(_)) for _ in range(support_num_user)],
    'support_item': [tf.placeholder(dtype=tf.float32, name='support'+str(_)) for _ in range(support_num_item)],
    'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
    'negative': tf.placeholder(dtype=tf.int32, shape=negative.shape, name='negative'),
    'batch_u': tf.placeholder(tf.int32, shape=(None, 1), name="user"),
    'batch_i': tf.placeholder(tf.int32, shape=(None, 1), name="item_pos"),
    'batch_j': tf.placeholder(tf.int32, shape=(None, 1), name="item_neg")
}
global_ = tf.Variable(tf.constant(0))
learning = tf.train.exponential_decay(learning_rate, global_, decay_steps, decay_rate, staircase=False)

# Create Model
model = MOOCUM(placeholders,
               input_dim_user=features_user.shape[1],
               input_dim_item=features_item.shape[1],
               user_dim=user_dim,
               item_dim=item_dim,
               learning_rate=learning)

# Initialize session
sess = tf.Session()

# Init variables
sess.run(tf.global_variables_initializer())

# Load from previous session
# model.load(sess=sess)

# ------------------------------------------
# Train model
start_time = time.time()
epoch = 0
mrr_best = 0
hrat5_best = 0
hrat10_best = 0
hrat20_best = 0
ndcgat5_best = 0
ndcgat10_best = 0
ndcgat20_best = 0

# Construct feed dictionary
feed_dict = construct_feed_dict(placeholders, features_user, features_item, rating, support_user,
                                support_item, negative)

total_params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
print("Total params of current model: {}".format(total_params))

while epoch < global_steps:
    for _ in range(batches):
        features = np.matmul(adjacency_matrix, features_item)
        features_user = features / adjacency_matrix.sum(axis=1) # avg of emb

        # Prepare batches
        # First we sample [samples] uniform indices
        idx = np.random.randint(low=0, high=len(uids), size=samples)
        # print("random sample indices:", idx[:10])
        # User batch matching idx
        batch_u = uids[idx].reshape(-1, 1)
        # Pos item
        batch_i = iids[idx].reshape(-1, 1)
        # Neg item
        batch_j = np.random.randint(
            low=0,
            high=item_dim,
            size=(samples, 1),
            dtype="int32"
        )
        # To feed, need to change dtype
        batch_u = batch_u.astype("float32")
        batch_i = batch_i.astype("float32")
        batch_j = batch_j.astype("float32")

        # Update feed_dict
        feed_dict.update({placeholders['features_user']: features_user})
        feed_dict.update({placeholders['features_item']: features_item})
        feed_dict.update({placeholders['batch_u']: batch_u})
        feed_dict.update({placeholders['batch_i']: batch_i})
        feed_dict.update({placeholders['batch_j']: batch_j})
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        feed_dict.update({global_: epoch})

        # Train with batch
        _, los, l2_los, alpha1, alpha2, HR1, HR5, HR10, HR20, NDCG5, NDCG10, NDCG20, MRR, AUC, user, item, result, \
        rate_matrix, rate1, rate2, bias, alphas_user, alphas_item = \
            sess.run([model.train_op, model.los, model.l2_loss,
                      model.layers[-1].vars['alpha1'], model.layers[-1].vars['alpha2'], model.hrat1,
                      model.hrat5, model.hrat10, model.hrat20,
                      model.ndcg5, model.ndcg10, model.ndcg20,
                      model.mrr, model.auc,
                      model.user, model.item,
                      model.result, model.rate_matrix,
                      model.rate1, model.rate2, model.bias,
                      # model.layers[-1].vars['alphas_user'], model.layers[-1].vars['alphas_item']], # for mfAtt
                      model.userModel.layers[-1].vars['alphas_user'], model.itemModel.layers[-1].vars['alphas_item']],
                     feed_dict)

    if epoch % 1 == 0:
        aLine = time.ctime() + \
                " {:10.2f}s passed".format(time.time()-start_time) + \
                " Train" + str(epoch) + \
                " Total-Loss:{:8.6f}".format(los) + \
                " L2-Loss:{:8.6f}".format(l2_los) + \
                " Model-Loss:{:8.6f}".format(los-l2_los) + \
                " HR@1:{:8.6f}".format(HR1) + \
                " HR@5:{:8.6f}".format(HR5) + \
                " HR@10:{:8.6f}".format(HR10) + \
                " HR@20:{:8.6f}".format(HR20) + \
                " nDCG@5:{:8.6f}".format(NDCG5) + \
                " nDCG@10:{:8.6f}".format(NDCG10) + \
                " nDCG@20:{:8.6f}".format(NDCG20) + \
                " MRR:{:8.6f}".format(MRR) + \
                " AUC:{:8.6f}".format(AUC) + \
                " Alpha1:{:8.5f}".format(alpha1) + \
                " Alpha2:{:8.5f}".format(alpha2) + \
                " rate:{}".format(rate_matrix[0][:5]) + \
                " bias:{}".format(bias[0][:5]) + \
                " rate1:{}".format(rate1[0][:5]) + \
                " rate2:{}".format(rate2[0][:5]) + \
                " alphas_user:{}".format(alphas_user[:, :5]) + \
                " alphas_item:{}".format(alphas_item[:, :5])
        print(aLine)
    epoch += 1

    # Save rating prediction with best performance
    if epoch > 100:
        # if HR5 > hrat5_best:
        #     print("Best HR5-{} updated at epoch:{}".format(HR5, epoch))
        #     hrat5_best = HR5
        #     with open('./output/m_rating_pred_besthr5.p', 'wb') as f:
        #         pkl.dump(rate_matrix, f)
        #     # Save
        #     model.save(sess, info="besthr5")
        #     np.save('./output/alphas_user_besthr5', alphas_user)
        #     np.save('./output/alphas_item_besthr5', alphas_user)
        # if HR10 > hrat10_best:
        #     print("Best HR10-{} updated at epoch:{}".format(HR10, epoch))
        #     hrat10_best = HR10
        #     with open('./output/m_rating_pred_besthr10.p', 'wb') as f:
        #         pkl.dump(rate_matrix, f)
        #     # Save
        #     model.save(sess, info="besthr10")
        #     np.save('./output/alphas_user_besthr10', alphas_user)
        #     np.save('./output/alphas_item_besthr10', alphas_user)
        # if HR20 > hrat20_best:
        #     print("Best HR20-{} updated at epoch:{}".format(HR20, epoch))
        #     hrat20_best = HR20
        #     with open('./output/m_rating_pred_besthr20.p', 'wb') as f:
        #         pkl.dump(rate_matrix, f)
        #     # Save
        #     model.save(sess, info="besthr20")
        #     np.save('./output/alphas_user_besthr20', alphas_user)
        #     np.save('./output/alphas_item_besthr20', alphas_user)
        # if NDCG5 > ndcgat5_best:
        #     print("Best NDCG5-{} updated at epoch:{}".format(NDCG5, epoch))
        #     ndcgat5_best = NDCG5
        #     with open('./output/m_rating_pred_bestndcg5.p', 'wb') as f:
        #         pkl.dump(rate_matrix, f)
        #     # Save
        #     model.save(sess, info="bestndcg5")
        #     np.save('./output/alphas_user_bestndcg5', alphas_user)
        #     np.save('./output/alphas_item_bestndcg5', alphas_user)
        # if NDCG10 > ndcgat10_best:
        #     print("Best NDCG10-{} updated at epoch:{}".format(NDCG10, epoch))
        #     ndcgat10_best = NDCG10
        #     with open('./output/m_rating_pred_bestndcg10.p', 'wb') as f:
        #         pkl.dump(rate_matrix, f)
        #     # Save
        #     model.save(sess, info="bestndcg10")
        #     np.save('./output/alphas_user_bestndcg10', alphas_user)
        #     np.save('./output/alphas_item_bestndcg10', alphas_user)
        # if NDCG20 > ndcgat20_best:
        #     print("Best NDCG20-{} updated at epoch:{}".format(NDCG20, epoch))
        #     ndcgat20_best = NDCG20
        #     with open('./output/m_rating_pred_bestndcg20.p', 'wb') as f:
        #         pkl.dump(rate_matrix, f)
        #     # Save
        #     model.save(sess, info="bestndcg20")
        #     np.save('./output/alphas_user_bestndcg20', alphas_user)
        #     np.save('./output/alphas_item_bestndcg20', alphas_user)
        if MRR > mrr_best:
            print("Best MRR-{} updated at epoch:{}".format(MRR, epoch))
            mrr_best = MRR
            with open('./output/m_rating_pred_bestmrr.p', 'wb') as f:
                pkl.dump(rate_matrix, f)
            # Save
            model.save(sess, info="bestmrr")
            np.save('./output/alphas_user_mrr', alphas_user)
            np.save('./output/alphas_item_mrr', alphas_user)

    # Save rating prediction every 50 epoch
    # if (epoch) % 50 == 0:
    #     with open('./output/m_rating_pred_ep{}.p'.format(epoch-1), 'wb') as f:
    #         pkl.dump(rate_matrix, f)
    #     # Save
    #     model.save(sess, info="ep{}".format(epoch))