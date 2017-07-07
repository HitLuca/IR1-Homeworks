import time
import math

import lasagne
import numpy as np
import theano
import theano.tensor as T

import query

import os

NUM_EPOCHS = 500

BATCH_SIZE = 1000
NUM_HIDDEN_UNITS = 100
LEARNING_RATE = 1 * pow(10, -8)
MOMENTUM = 0.95


def lambda_loss(output, lambdas):
    """loss used by pairwise and listwise models, uses a Theano workaround that allows us to output a loss and derive
    the correct gradients when updating the neural netowrk parameters"""
    return T.dot(output, lambdas)


class ResultWriter:
    """Simple class that saves all the results to file"""

    def __init__(self, filepath):
        self.file = open(filepath, 'w')
        self.write_header()

    def write_epoch(self, epoch_number, elapsed_time, loss, train_ndcg, val_ndcg):
        self.file.write(
            str(epoch_number) + ', ' + str(elapsed_time) + ', ' + str(loss) + ', ' + str(train_ndcg) + ', ' + str(
                val_ndcg) + '\n')
        self.file.flush()

    def write_test_score(self, ndcg):
        self.file.write('\n' + str(ndcg) + '\n')
        self.file.flush()

    def finalize(self):
        self.file.close()

    def write_header(self):
        self.file.write('epoch number, elapsed time, train loss, train ndcg@10, val ndcg@10' + '\n')
        self.file.flush()


class LambdaRankHW:
    def __init__(self, feature_count, method, results_filepath):
        self.feature_count = feature_count
        self.method = method
        self.output_layer = self.build_model(feature_count, 1, BATCH_SIZE)
        self.iter_funcs = self.create_functions(self.output_layer)
        self.result_writer = ResultWriter(results_filepath)

    def train_with_queries(self, train_queries, val_queries, test_queries, epochs=-1):
        """If no epoch number are defined then Early Stopping is used, otherwise the old iterative model is used"""

        try:
            if epochs != -1:
                for i in range(epochs):
                    epoch_number = i + 1
                    now = time.time()
                    epoch = self.train(train_queries)
                    elapsed_time = time.time() - now

                    loss = "{:.6f}".format(epoch['train_loss'])
                    ndcg = "{:.6f}".format(epoch['ndcg@10'])

                    print("Epoch {} of {} took {:.3f}s".format(
                        epoch_number, epochs, elapsed_time))
                    print("Training loss:\t\t" + loss)
                    print("Training ndcg@10:\t\t" + ndcg + "\n")

                    self.result_writer.write_epoch(epoch_number, elapsed_time, loss, ndcg, '')
                test_score = self.evaluate_queries(test_queries)
                print('Test score: ' + "{:.6f}".format(test_score))
                self.result_writer.write_test_score(test_score)
            else:
                """past_scores is a sliding window of size 5 that stores the last 5 validation scores in order to check
                 if the stopping requiriments are met"""
                past_scores = [0 for _ in range(5)]  # N scores sliding window for early stopping
                val_ndcg = 0
                epoch_number = 1
                while min(past_scores) <= val_ndcg:
                    past_scores.pop(0)
                    past_scores.append(val_ndcg)

                    now = time.time()
                    epoch = self.train(train_queries)
                    elapsed_time = time.time() - now

                    loss = "{:.6f}".format(epoch['train_loss'])
                    ndcg = "{:.6f}".format(epoch['ndcg@10'])
                    val_ndcg = "{:.6f}".format(self.evaluate_queries(val_queries))

                    print("Epoch {} took {:.3f}s".format(
                        epoch_number, elapsed_time))
                    print("Training loss:\t\t" + loss)
                    print("Training ndcg@10:\t" + ndcg)
                    print('Validation ndcg@10:\t' + val_ndcg + '\n')
                    self.result_writer.write_epoch(epoch_number, elapsed_time, loss, ndcg, val_ndcg)
                    val_ndcg = float(val_ndcg)
                    epoch_number += 1
                test_score = self.evaluate_queries(test_queries)
                print('Test score: ' + "{:.6f}".format(test_score))
                self.result_writer.write_test_score(test_score)
            return test_score
        except KeyboardInterrupt:
            pass

    def score(self, query):
        feature_vectors = query.get_feature_vectors()
        scores = self.iter_funcs['out'](feature_vectors)
        return scores

    @staticmethod
    def build_model(input_dim, output_dim, batch_size=BATCH_SIZE):
        l_in = lasagne.layers.InputLayer(
            shape=(batch_size, input_dim),
        )

        l_hidden = lasagne.layers.DenseLayer(
            l_in,
            num_units=200,
            nonlinearity=lasagne.nonlinearities.tanh,
        )

        l_out = lasagne.layers.DenseLayer(
            l_hidden,
            num_units=output_dim,
            nonlinearity=lasagne.nonlinearities.linear,
        )

        return l_out

    def create_functions(self, output_layer, X_tensor_type=T.matrix, L1_reg=0.0000005, L2_reg=0.000003):
        """Create functions for training, validation and testing to iterate one
           epoch."""
        X_batch = X_tensor_type('x')
        y_batch = T.fvector('y')

        output_row = lasagne.layers.get_output(output_layer, X_batch, dtype="float32")
        output = output_row.T

        output_row_det = lasagne.layers.get_output(output_layer, X_batch, deterministic=True, dtype="float32")

        if self.method == 0:
            loss_train = lasagne.objectives.squared_error(output, y_batch)

        elif self.method == 1 or self.method == 2:
            loss_train = lambda_loss(output, y_batch)

        loss_train = loss_train.mean()

        L1_loss = lasagne.regularization.regularize_network_params(output_layer, lasagne.regularization.l1)
        L2_loss = lasagne.regularization.regularize_network_params(output_layer, lasagne.regularization.l2)
        loss_train = loss_train + L1_loss * L1_reg + L2_loss * L2_reg

        # Parameters you want to update
        all_params = lasagne.layers.get_all_params(output_layer)

        # Update parameters, adam is a particular "flavor" of Gradient Descent
        updates = lasagne.updates.adam(loss_train, all_params)

        score_func = theano.function(
            [X_batch], output_row_det,
        )

        train_func = theano.function(
            [X_batch, y_batch], loss_train,
            updates=updates,
            allow_input_downcast=True  # In order to use simple float32 Numpy arrays
        )

        return dict(
            train=train_func,
            out=score_func,
        )

    def calculate_ndcg_10(self, labels, query):
        """This is a function that calculates the ndgc@10, for a given query and the corresponding output ranking"""
        scores = self.score(query).flatten()
        indices = np.argsort(scores)[-10:][::-1]
        relevant = [i for i, x in enumerate(labels) if x == 1]

        dcg_i = 0
        idcg_i = 0
        for j in range(10):
            dcg_i += (pow(2, labels[indices[j]]) - 1) / (math.log(j + 2, 2))

        for j in range(len(relevant)):
            idcg_i += 1 / (math.log(j + 2, 2))

        if idcg_i != 0:
            return dcg_i / idcg_i
        return 0

    @staticmethod
    def calculate_ndcg_matrix(labels, relevant, irrelevant, scores):
        """This function is only used for the LambdaRank model, and calculates the delta_NDCG@MAX"""

        """Calculation of the IDCG, in order to normalize the delta_NDCG@MAX"""
        idcg_i = 0
        for j in range(len(relevant)):
            idcg_i += 1 / (math.log(j + 2, 2))

        """Calculation of each delta_NDCG@MAX, which can be simplified to the used formula, since
        all other rankings won't contribute to the final difference"""
        matrix = np.zeros((len(relevant), len(irrelevant)))
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                matrix[i][j] = abs((pow(2, labels[relevant[i]]) - pow(2, labels[irrelevant[j]])) * (
                    1 / math.log(i + 2, 2) - 1 / math.log(j + 2, 2)))

        matrix /= idcg_i
        return matrix

    def lambda_function(self, labels, scores):
        aggregated = np.empty(len(labels))

        """Determines the documents that are actually relevant/irrelevant for the query in question"""
        relevant = [i for i, x in enumerate(labels) if x == 1]
        irrelevant = [i for i, x in enumerate(labels) if x == 0]

        """Creates a matrix that will contain only the relevant lambdas, that is, where doc_i and
        doc_j have different relevances"""
        matrix = np.empty((len(relevant), len(irrelevant)))

        """Calculates the lambda values for each document pair where doc_i and
        doc_j have different relevances"""
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                matrix[i][j] = scores[relevant[i]] - scores[irrelevant[j]]
        matrix = -1.0 / (1 + np.exp(matrix))

        """Calculates the \delta_NDCG@MAX for each corresponding lambda value, when using LambdaRank
        and updates the lambda values for each pair with the corresponding \delta_NDCG@MAX"""
        if self.method == 2:
            matrix *= self.calculate_ndcg_matrix(labels, relevant, irrelevant, scores)

        """Calculates de aggregated lambdas for each relevant document"""
        sum_relevant = np.sum(matrix, axis=1)
        for i, index in enumerate(relevant):
            aggregated[index] = sum_relevant[i]

        """Calculates de aggregated lambdas for each irrelevant document"""
        sum_irrelevant = np.sum(matrix, axis=0)
        for i, index in enumerate(irrelevant):
            aggregated[index] = - sum_irrelevant[i]

        return aggregated

    def compute_lambdas_theano(self, query, labels):
        scores = self.score(query).flatten()
        result = self.lambda_function(labels, scores[:len(labels)])
        return result

    def train_once(self, X_train, query, labels):
        """The changes made to this function are simple and only relevant for model selection,
        calculating the ndcg@10 and calling the lambdas computation functions"""

        if self.method == 1 or self.method == 2:
            lambdas = self.compute_lambdas_theano(query, labels)
            lambdas.resize((BATCH_SIZE,))
            X_train.resize((min(BATCH_SIZE, len(labels)), self.feature_count), refcheck=False)

        if self.method == 0:
            batch_train_loss = self.iter_funcs['train'](X_train, labels)
        elif self.method == 1 or self.method == 2:
            batch_train_loss = self.iter_funcs['train'](X_train, lambdas)

        ndcg = self.calculate_ndcg_10(labels, query)

        return batch_train_loss, ndcg

    def train(self, train_queries):
        """The changes made to this function are simple and only relevant for calculating the
        average ndcg@10 and train losses"""
        X_trains = train_queries.get_feature_vectors()

        queries_values = train_queries.values()

        batch_train_losses = []
        batch_ndcgs = []
        random_batch = np.arange(len(queries_values))
        np.random.shuffle(random_batch)
        for index in xrange(len(queries_values)):
            random_index = random_batch[index]
            labels = queries_values[random_index].get_labels()

            batch_train_loss, batch_ndcg = self.train_once(X_trains[random_index], queries_values[random_index],
                                                           labels)
            if batch_ndcg != 0:
                batch_ndcgs.append(batch_ndcg)
            batch_train_losses.append(batch_train_loss)

        avg_train_loss = np.mean(batch_train_losses)
        avg_ndcg = np.mean(batch_ndcgs)

        return {
            'train_loss': avg_train_loss,
            'ndcg@10': avg_ndcg
        }

    def evaluate_queries(self, queries):
        """This function evaluates a given set of queries, in order to retrieve validation and testing scores"""

        ndcgs = []
        for query in queries:
            ndcg = self.calculate_ndcg_10(query.get_labels(), query)
            if ndcg != 0:
                ndcgs.append(ndcg)
        return np.mean(ndcgs)


if not os.path.exists('results'):
    os.makedirs('results')

method_scores = []

folds = 5

# Model identification number
# 0: pointwise
# 1: pairwise
# 2: listwise
"""This code loops over each model type, and for each model type over all folds, retrieving the final scores"""
for m in range(3):
    print('Method ' + str(m))
    if not os.path.exists('results/method' + str(m)):
        os.makedirs('results/method' + str(m))

    test_scores = []
    for i in range(folds):
        fold = i + 1
        print('Fold ' + str(fold))
        print('Creating model')
        model = LambdaRankHW(64, 2, 'results/' + 'method' + str(m) + '/results' + str(fold) + '.csv')
        print('Model created')
        print('Loading queries')
        train_queries = query.load_queries('HP2003/Fold' + str(fold) + '/train.txt', 64)
        val_queries = query.load_queries('HP2003/Fold' + str(fold) + '/vali.txt', 64)
        test_queries = query.load_queries('HP2003/Fold' + str(fold) + '/test.txt', 64)
        print('Loaded\n')

        test_scores.append(model.train_with_queries(train_queries, val_queries, test_queries, epochs=15))
        print('\n')
    method_scores.append(np.mean(test_scores))
print('Final scores: ')
print(method_scores)
