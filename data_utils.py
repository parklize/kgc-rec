import pickle as pkl
import numpy as np
from sklearn.metrics import ndcg_score
np.random.seed(0)


def evaluate(pred_rating_matrix, training_matrix, ground_truth_matrix, user_specific=False, for_csv=False):
    """ 
    Prepare negative item indices to sample 99 items for each positive in testing
    For each user get testing items,
        For each testing item, paring with 99 negative samples
            Calculate relevant eval metric
    Average the evaluation score over all users as defined in the paper

    Parameters
    -----------------------
    user_specific: mean of user specific average scores (scores for multiple items of a user is averaged)
    for_csv: only print csv files if True for dataframe analysis
    """
    user_size = pred_rating_matrix.shape[0]
    item_size = pred_rating_matrix.shape[1]
    item_indices = list(range(item_size))
    # print("user size:{}, item size:{}".format(user_size, item_size))

    # Get negative item indices
    # Get testing items (ground truth - training)
    hr5 = 0.
    hr10 = 0.
    hr20 = 0.
    mrr = 0.
    ndcg5 = 0.
    ndcg10 = 0.
    ndcg20 = 0.
    valid_testing_size = 0
    valid_testing_pairs = list() # how many 100 (one pos, 99 neg) pairs tested
    for i in range(user_size):
        # per user
        hr5_ = 0.
        hr10_ = 0.
        hr20_ = 0.
        mrr_ = 0.
        ndcg5_ = 0.
        ndcg10_ = 0.
        ndcg20_ = 0.
        # if i % 50 == 0:
        #     print("{}-th user".format(i))
        # negative indices for the current user
        # neg_indices = list(set(item_indices)-set(ground_truth_matrix[i]))
        # neg_indices = [_ for _ in item_indices if _ not in np.where(ground_truth_matrix[i]>0)[0]]
        neg_indices = np.where(ground_truth_matrix[i] == 0)[0]

        # testing indices for the current user
        training_indices = np.where(training_matrix[i] > 0)[0]
        testing_indices = [x for x in np.where(ground_truth_matrix[i] > 0)[0] if x not in training_indices]
        # print("testing indices:{}".format(testing_indices))
        if len(testing_indices) > 0:
            for ti in testing_indices:
                # get testing size*99 items from neg_indices
                selected_neg_indices = np.random.choice(np.array(neg_indices), 99, replace=False)
                # print(selected_neg_indices)
                indices = np.array(list(selected_neg_indices)+[ti])
                # valid_testing_pairs.append(np.stack((np.array([i] * 100), indices), axis=1))
                indices_sorted = np.argsort(pred_rating_matrix[i][indices])
                # print(valid_testing_size, pred_rating_matrix[i][indices])
                # print(indices_sorted)
                ground_truth_indices = [99]
                # ground_truth_indices = list(range(len(indices)-1, len(indices)-1-len(testing_indices), -1))
                hr5_ex = 1. if len(intersection(ground_truth_indices, indices_sorted[-5:])) > 0 else 0.
                hr5_ += hr5_ex
                hr10_ex = 1. if len(intersection(ground_truth_indices, indices_sorted[-10:])) > 0 else 0.
                hr10_ += hr10_ex
                hr20_ex = 1. if len(intersection(ground_truth_indices, indices_sorted[-20:])) > 0 else 0.
                hr20_ += hr20_ex
                index = np.max([np.where(indices_sorted == ind) for ind in ground_truth_indices][0][0])
                # sorted is reversed rank for mrr
                rank = len(indices_sorted) - index
                mrr_ex = (1./rank)
                mrr_ += mrr_ex
                # NDCG@K
                y_true = np.asarray([[0.]*len(selected_neg_indices)+[1]])
                y_pred = np.asarray([pred_rating_matrix[i][indices]])
                # print(y_true.shape, y_pred.shape)
                ndcg5_ex = ndcg_score(y_true, y_pred, k=5)
                ndcg5_ += ndcg5_ex
                ndcg10_ex = ndcg_score(y_true, y_pred, k=10)
                ndcg10_ += ndcg10_ex
                ndcg20_ex = ndcg_score(y_true, y_pred, k=20)
                ndcg20_ += ndcg20_ex

                valid_testing_size += 1

                print(i, len(training_indices), len(testing_indices),
                      hr5_ex, hr10_ex, hr20_ex, ndcg5_ex, ndcg10_ex, ndcg20_ex, mrr_ex)

            if user_specific:
                num_examples = float(len(testing_indices))
                hr5 += hr5_/num_examples
                hr10 += hr10_/num_examples
                hr20 += hr20_/num_examples
                ndcg5 += ndcg5_/num_examples
                ndcg10 += ndcg10_/num_examples
                ndcg20 += ndcg20_/num_examples
                mrr += mrr_/num_examples
            else:
                hr5 += hr5_
                hr10 += hr10_
                hr20 += hr20_
                ndcg5 += ndcg5_
                ndcg10 += ndcg10_
                ndcg20 += ndcg20_
                mrr += mrr_

    # Store testing pairs
    # np.save("testing_pairs", np.array(valid_testing_pairs), allow_pickle=False)
    if not for_csv:
        if user_specific:
            print("valid testing size:{}".format(user_size))
            print(
                "hr@5:{:7.4f} hr@10:{:7.4f} hr@20:{:7.4f} mrr:{:7.4f} ndcg@5:{:7.4f} ndcg@10:{:7.4f} ndcg@20:{:7.4f}".format(
                    hr5 / user_size,
                    hr10 / user_size,
                    hr20 / user_size,
                    mrr / user_size,
                    ndcg5 / user_size,
                    ndcg10 / user_size,
                    ndcg20 / user_size
                ))
        else:
            print("valid testing size:{}".format(valid_testing_size))
            print("hr@5:{:7.4f} hr@10:{:7.4f} hr@20:{:7.4f} mrr:{:7.4f} ndcg@5:{:7.4f} ndcg@10:{:7.4f} ndcg@20:{:7.4f}".format(
                hr5/valid_testing_size,
                hr10/valid_testing_size,
                hr20/valid_testing_size,
                mrr/valid_testing_size,
                ndcg5/valid_testing_size,
                ndcg10/valid_testing_size,
                ndcg20/valid_testing_size
            ))


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


if __name__ == "__main__":
    # Ground truth
    training_matrix_f = './data/adjacency_matrix.p'
    ground_truth_matrix_f = './data/user_action.p'
    # Prediction file
    pred_matrix_f = './output/m_rating_pred_bestmrr.p'

    # -------------------------------------------
    # Load matrix
    with open(training_matrix_f, 'rb') as f:
        training_matrix = pkl.load(f)
        if not isinstance(training_matrix, np.matrix):
            training_matrix = training_matrix.todense()
        else:
            training_matrix = np.array(training_matrix)
        # np.save("training_matrix", training_matrix, allow_pickle=False)
    # with open("./MOOCCube/data-for-kgcrec/negative.p", 'rb') as f:
    #     negative = pkl.load(f)
    #     np.save("negative", negative, allow_pickle=False)
    with open(ground_truth_matrix_f, 'rb') as f:
        ground_truth_matrix = pkl.load(f)
        if not isinstance(ground_truth_matrix, np.matrix):
            ground_truth_matrix = ground_truth_matrix.todense()
        else:
            ground_truth_matrix = np.array(ground_truth_matrix)
    with open(pred_matrix_f, 'rb') as f:
        pred_matrix = pkl.load(f)
        # print(pred_matrix)

    # Evaluation
    evaluate(pred_matrix, training_matrix, ground_truth_matrix, user_specific=False, for_csv=False)
