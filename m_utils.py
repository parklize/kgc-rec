import numpy as np
import pickle as pkl
import random
import scipy.sparse as sp
import json
import tabulate
import time

DATA_FOLDER = "./data" # my prepared dataset from MOOCCube


def load_data(user, item):
    support_user = []
    support_item = []
    # rating matrix
    with open(DATA_FOLDER+'/rate_matrix.p', 'rb') as source:
        rating = pkl.load(source)
        if not isinstance(rating, np.matrix):
            rating = rating.todense()
    rating = np.log1p(rating)
    print("rating", rating)
    # concept w2v features (dim=100)
    with open(DATA_FOLDER+'/concept_embedding.p', 'rb') as source:
        concept_w2v = np.array(pkl.load(source))
    print("concept_w2v", concept_w2v)
    # concept bow features (dim=616)
    # with open(DATA_FOLDER+'/concept_feature_bow.p', 'rb') as source:
    #     concept_bow = pkl.load(source).todense()
    # concept = np.hstack((concept_w2v, concept_bow))
    concept = concept_w2v
    # dim=100+616=716
    # features_item = preprocess_features(concept.astype(np.float32))
    features_item = concept

    # user features (2136, 2844) total sum = 2136 - user-course relationship?
    # with open(DATA_FOLDER+'/UC.p', 'rb') as source:
    #     features = pkl.load(source)
    #     if not isinstance(features, np.matrix):
    #         features = features.todense()
    # features_user = preprocess_features(features.astype(np.float32))
    # I will change it to avg. of learned concept embeddings for each user
    features = np.matmul(rating, concept)
    features = features / rating.sum(axis=1) # avg
    features_user = preprocess_features(features.astype(np.float32))
    print("features_user", features_user)

    # uku - user-concept-user relationship
    if 'uku' in user or 'kuk' in item:
        with open(DATA_FOLDER+'/adjacency_matrix.p', 'rb') as source:
            adjacency_matrix = pkl.load(source)
            if not isinstance(adjacency_matrix, np.matrix):
                adjacency_matrix = adjacency_matrix.todense()

        # to float will speed up a lot
        uk = adjacency_matrix.astype(np.float32)
        # stime = time.time()
        ku = uk.T
        # print("transposed uk in {}s".format(time.time()-stime))
        if 'uku' in user: # user-user via concept
            stime = time.time()
            # uk_user = uk.dot(uk.T) + np.eye(uk.shape[0])
            uk_user = uk.dot(ku) + np.eye(uk.shape[0])
            uku = preprocess_adjacency(uk_user, name="uku")
            support_user.append(uku)
            print("added uku in {}s".format(time.time()-stime))
            print(uku)
        if 'kuk' in item: # concept-concept via user
            stime = time.time()
            # ku_item = uk.T.dot(uk) + np.eye(uk.T.shape[0])
            ku_item = ku.dot(uk) + np.eye(ku.shape[0])
            kuk = preprocess_adjacency(ku_item, name="kuk")
            support_item.append(kuk)
            print("added kuk in {}s".format(time.time()-stime))
            print(kuk)

    # ucu # user-user via course
    stime = time.time()
    if 'ucu' in user:
        with open(DATA_FOLDER+'/user_course.p', 'rb') as source:
            uc = pkl.load(source)
            if not isinstance(uc, np.matrix):
                uc = uc.todense()
        uc = uc.dot(uc.T) + np.eye(uc.shape[0])
        ucu = preprocess_adjacency(uc, name="ucu")
        support_user.append(ucu)
        print("added ucu in {}s".format(time.time()-stime))
        print(ucu)

    # uctcu # user-user via teacher
    stime = time.time()
    if 'uctcu' in user:
        with open(DATA_FOLDER+'/user_course_teacher.p', 'rb') as source:
            uct = pkl.load(source)
            if not isinstance(uct, np.matrix):
                uct = uct.todense()
        uct = uct.dot(uct.T) + np.eye(uct.shape[0])
        uctcu = preprocess_adjacency(uct, name="uctcu")
        support_user.append(uctcu)
        print("added uctcu in {}s".format(time.time()-stime))
        print(uctcu)

    # uvu # user-user via video
    stime = time.time()
    if 'uvu' in user:
        with open(DATA_FOLDER+'/user_video.p', 'rb') as source:
            uv = pkl.load(source)
            if not isinstance(uv, np.matrix):
                uv = uv.todense()
        uv = uv.dot(uv.T) + np.eye(uv.shape[0])
        uvu = preprocess_adjacency(uv, name="uvu")
        support_user.append(uvu)
        print('added uvu in {}s'.format(time.time()-stime))
        print(uvu)

    # kck # concept-course-concept relationship
    stime = time.time()
    if 'kck' in item:
        with open(DATA_FOLDER+'/concept_course.p', 'rb') as source:
            kc = pkl.load(source)
            if not isinstance(kc, np.matrix):
                kc = kc.todense()
        kc = kc.astype(np.float32) # to speed up
        kc = kc.dot(kc.T) + np.eye(kc.shape[0])
        kck = preprocess_adjacency(kc, name="kck")
        support_item.append(kck)
        print("added kck in {}s".format(time.time()-stime))
        print(kck)

    support_user = np.array(support_user)
    support_item = np.array(support_item)
    print(support_user.shape)
    print(support_item.shape)
    # assert len(item) == support_item.shape[0]
    # assert len(user) == support_user.shape[0]

    # todo dummy input when testing MF only
    # support_item = np.array([np.zeros((21037, 21037))])
    # print(support_item.shape)

    # print("support_user", support_user)
    # print("support_item", support_item)

    # negative sample
    with open(DATA_FOLDER+'/negative.p', 'rb') as source:
        negative = pkl.load(source)
    # print("negative", negative)

    return rating, adjacency_matrix, features_item, features_user, support_user, support_item, negative


# def get_user_emb(concepts, concept_emb):
#

def preprocess_features(features):
    """ Preprocess to make row sum as prob. i.e., sum=1 """
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    features = r_mat_inv.dot(features)
    return features


def preprocess_adj(adjacency, name=""):
    rowsum = np.array(adjacency.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    print("Degree matrix-0.5 power for {} - {}".format(name, d_inv_sqrt))
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adjacency.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)*1e2


def preprocess_adjacency(A, name=""):
    """
    https://towardsdatascience.com/how-to-do-deep-learning-on-graphs-with-graph-convolutional-networks-7d2250723780
    2 different normalization where one is preprocess_adj - https://tkipf.github.io/graph-convolutional-networks/
    """
    D = np.array(np.sum(A, axis=0)).flatten()
    print("Degree matrix for {} - {}".format(name, D))
    D = np.matrix(np.diag(D))
    return D**-1 * A * 1e2


def construct_feed_dict(placeholders, features_user, features_item, rating, biases_list_user,
                        biases_list_item, negative):
    feed_dict = dict()
    feed_dict.update({placeholders['rating']: rating})
    feed_dict.update({placeholders['features_user']: features_user})
    feed_dict.update({placeholders['features_item']: features_item})
    feed_dict.update({placeholders['support_user'][i]: biases_list_user[i] for i in range(len(biases_list_user))})
    feed_dict.update({placeholders['support_item'][i]: biases_list_item[i] for i in range(len(biases_list_item))})
    feed_dict.update({placeholders['negative']: negative})
    return feed_dict


def radom_negative_sample(user_action, item_size):
    negative_sample = []
    for u in user_action:
        sample = []
        i = 0
        while i < 99:
            t = random.randint(0, item_size-1)
            if t not in user_action[u]:
                sample.append([u, t])
                i += 1
        sample.append([u, user_action[u][-1]])
        negative_sample.append(sample)
    return np.array(negative_sample)


def getRateMatrix(user_action, item_size):
    """ Get rate matrix where rate = # of clicks of a concept """
    row = []
    col = []
    dat = []
    for u in user_action:
        ls = set(user_action[u])
        for k in ls:
            row.append(u)
            col.append(k)
            dat.append(user_action[u].count(k))
    coo_matrix = sp.coo_matrix((dat, (row, col)), shape=(len(user_action), item_size))
    with open('./data/rate_matrix_new.p', 'wb') as source:
        pkl.dump(coo_matrix.toarray(), source)


def adj_to_bias(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):

            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0