import os
import math
import time
import yaml
import uuid
import pickle
import logging
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from ast import literal_eval as make_tuple

import utils
from dataset import DataGenerator
from projector_plugin import ProjectorPlugin

#convert all print's to log

# PUT THESE UNDER CONFIG.YML
# project-wide constants:
ROOT_LOGGER_STR = "DCGMM"
LOGGER_RESULT_FILE = "logs.txt"
CHECKPOINT_PATH = 'models'  # "autoencoder/cp.ckpt"

#CREATE THIS WITHIN MAIN no printing anymore
logger = logging.getLogger(ROOT_LOGGER_STR + '.' + __name__)


    
# PUT UNDER UTILS.PY
def loss_DCGMM_mnist(inp, x_decoded_mean):
    x = inp
    loss = 784 * tf.keras.losses.BinaryCrossentropy()(x, x_decoded_mean)
    return loss

# PUT UNDER UTILS.PY
def accuracy_metric(inp, p_c_z):
    y = inp
    y_pred = tf.math.argmax(p_c_z, axis=-1)
    return tf.numpy_function(utils.cluster_acc, [y, y_pred], tf.float64)


# GET PRETRAIN MODEL FROM UTILS.PY
def pretrain(model, args, ex_name, configs):
    input_shape = configs['training']['inp_shape']
    num_clusters = configs['training']['num_clusters']

    if configs['data']['data_name'] in ["heart_echo", "cifar10", "utkface"]:
        if configs['training']['type'] in ["CNN", "VGG"]:
            input_shape = make_tuple(input_shape)

    # Get the AE from the model
    input = tfkl.Input(shape=input_shape)

    if configs['training']['type'] == "FC":
        f = tfkl.Flatten()(input)
        e1 = model.encoder.dense1(f)
        e2 = model.encoder.dense2(e1)
        e3 = model.encoder.dense3(e2)
        z = model.encoder.mu(e3)
        d1 = model.decoder.dense1(z)
        d2 = model.decoder.dense2(d1)
        d3 = model.decoder.dense3(d2)
        dec = model.decoder.dense4(d3)
    elif configs['training']['type'] == "CNN":
        e1 = model.encoder.conv1(input)
        e2 = model.encoder.conv2(e1)
        f = tfkl.Flatten()(e2)
        z = model.encoder.mu(f)
        d1 = model.decoder.dense(z)
        d2 = model.decoder.reshape(d1)
        d3 = model.decoder.convT1(d2)
        d4 = model.decoder.convT2(d3)
        d5 = model.decoder.convT3(d4)
        dec = tf.sigmoid(d5)
    elif configs['training']['type'] == "VGG":
        enc = input
        for block in model.encoder.layers:
            enc = block(enc)
        f = tfkl.Flatten()(enc)
        z = model.encoder.mu(f)
        d_dense = model.decoder.dense(z)
        d_reshape = model.decoder.reshape(d_dense)
        dec = d_reshape
        for block in model.decoder.layers:
            dec = block(dec)
        dec = model.decoder.convT(dec)
        dec = tf.sigmoid(dec)

    autoencoder = tfk.Model(inputs=input, outputs=dec)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # , decay=args.decay)
    if args.data == 'MNIST' or args.data == 'fMNIST' or args.data == 'heart_echo' or args.data == 'utkface':
        autoencoder.compile(optimizer=optimizer, loss="binary_crossentropy")
    else:
        autoencoder.compile(optimizer=optimizer, loss="mse")
    autoencoder.summary()
    x_train, x_test, y_train, y_test = utils.get_data(args, configs)
    X = np.concatenate((x_train, x_test))
    Y = np.concatenate((y_train, y_test))

    # If the model should be run from scratch:
    if args.pretrain:
        #print('\n******************** Pretraining **************************')
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="pretrain/autoencoder_tmp/" + ex_name + "/cp.ckpt",
                                                         save_weights_only=True, verbose=1)
        autoencoder.fit(X, X, epochs=args.epochs_pretrain, batch_size=32, callbacks=cp_callback)

        encoder = model.encoder
        input = tfkl.Input(shape=input_shape)
        z, _ = encoder(input)
        z_model = tf.keras.models.Model(inputs=input, outputs=z)
        z = z_model.predict(X)

        estimator = GaussianMixture(n_components=num_clusters, covariance_type='diag', n_init=3)
        estimator.fit(z)
        pickle.dump(estimator, open("pretrain/gmm_tmp/" + ex_name + "_gmm_save.sav", 'wb'))

        #print('\n******************** Pretraining Done**************************')
    else:
        if args.data == 'MNIST':
            autoencoder.load_weights("pretrain/MNIST/autoencoder/cp.ckpt")
            estimator = pickle.load(open("pretrain/MNIST/gmm_save.sav", 'rb'))
            #print('\n******************** Loaded MNIST Pretrain Weights **************************')
        else:
            print('\nPretrained weights for {} not available, please rerun with \'--pretrain True option\''.format(
                args.data))
            exit(1)
    
    encoder = model.encoder
    input = tfkl.Input(shape=input_shape)
    z, _ = encoder(input)
    z_model = tf.keras.models.Model(inputs=input, outputs=z)

    # Assign weights to GMM mixtures of DCGMM
    mu_samples = estimator.means_
    sigma_samples = estimator.covariances_
    model.c_mu.assign(mu_samples)
    model.log_c_sigma.assign(np.log(sigma_samples))

    yy = estimator.predict(z_model.predict(X))
    acc = utils.cluster_acc(yy, Y)
    pretrain_acc = acc
    print('\nPretrain accuracy: ' + str(acc))

    return model, pretrain_acc


def run_experiment(cfg):#args, configs, loss):
    
    if cfg['experiment']['name'] is None:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        ex_name = "{}_{}".format(str(timestr), uuid.uuid4().hex[:5])
    else:
        ex_name = cfg['experiment']['name']

    experiment_path = Path(os.path.join(cfg['dir']['logging'], cfg['dataset']['name'], ex_name))
    experiment_path.mkdir(parents=True)
    
    
    x_train, x_test, y_train, y_test = utils.get_data(cfg)

    acc_tot, nmi_tot, ari_tot = [], [], []

    model = utils.get_model(cfg)

    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg['learning_rate'], decay=cfg['decay'])

    # get from utils
    # load all callbacks from utils.
    def learning_rate_scheduler(epoch):
        initial_lrate = args.lr
        drop = args.decay_rate
        epochs_drop = args.epochs_lr
        lrate = initial_lrate * math.pow(drop,
                                            math.floor((1 + epoch) / epochs_drop))
        return lrate

    if cfg['lrs']:
        cp_callback = [tf.keras.callbacks.TensorBoard(log_dir='logs/' + ex_name),
                        tf.keras.callbacks.LearningRateScheduler(learning_rate_scheduler)]
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    elif cfg['save_model']:
        checkpoint_path = CHECKPOINT_PATH + '/' + configs['data']['data_name'] + '/' + ex_name
        cp_callback = [tf.keras.callbacks.TensorBoard(log_dir='logs/' + ex_name),
                        tf.keras.callbacks.ModelCheckpoint(
                            filepath=checkpoint_path,
                            verbose=1,
                            save_weights_only=True,
                            period=100)]
    else:
        cp_callback = [tf.keras.callbacks.TensorBoard(log_dir='logs/' + ex_name)]

    model.compile(optimizer, loss={"output_1": loss}, metrics={"output_4": accuracy_metric})

    # pretrain model
    model, pretrain_acc = pretrain(model, args, ex_name, configs)

    # get alpha from utils.py
    if args.q > 0:
        alpha = 1000 * np.log((1 - args.q) / args.q)
    else:
        alpha = args.alpha

    # create data generators LOAD FROM UTILS.PY
    # utils.get_data_generator():
    gen = DataGenerator(x_train, y_train, num_constrains=args.num_constrains, alpha=alpha, q=args.q,
                        batch_size=args.batch_size, ml=args.ml)
    test_gen = DataGenerator(x_test, y_test, batch_size=args.batch_size).gen()

    train_gen = gen.gen()
    
    # fit model
    model.fit(train_gen, validation_data=test_gen, steps_per_epoch=int(len(y_train)/args.batch_size), validation_steps=len(y_test)//args.batch_size, epochs=args.num_epochs, callbacks=cp_callback)

    # results
    rec, z_sample, p_z_c, p_c_z = model.predict([x_train, np.zeros(len(x_train))])
    yy = np.argmax(p_c_z, axis=-1)
    acc = utils.cluster_acc(y_train, yy)
    nmi = normalized_mutual_info_score(y_train, yy)
    ari = adjusted_rand_score(y_train, yy)
    ml_ind1 = gen.ml_ind1
    ml_ind2 = gen.ml_ind2
    cl_ind1 = gen.cl_ind1
    cl_ind2 = gen.cl_ind2
    count = 0
    if args.num_constrains == 0:
        sc = 0
    else:
        maxx = len(ml_ind1) + len(cl_ind1)
        for i in range(len(ml_ind1)):
            if yy[ml_ind1[i]] == yy[ml_ind2[i]]:
                count += 1
        for i in range(len(cl_ind1)):
            if yy[cl_ind1[i]] != yy[cl_ind2[i]]:
                count += 1
        sc = count / maxx

    # no need to this. just log within the experiment file!!
    if args.data == 'MNIST':
        f = open("results_MNIST.txt", "a+")
    elif args.data == 'fMNIST':
        f = open("results_fMNIST.txt", "a+")
    elif args.data == 'Reuters':
        f = open("results_reuters.txt", "a+")
    elif args.data == 'heart_echo':
        f = open("results_heart_echo.txt", "a+")
        f.write("%s, %s. " % (configs['data']['label'], configs['training']['type']))
    elif args.data == 'stl10':
        f = open("results_stl.txt", "a+")
    elif args.data == 'utkface':
        f = open("results_utkface.txt", "a+")
        f.write("%s. " % (configs['data']['label']))
    f.write("Epochs= %d, num_constrains= %d, ml= %d, alpha= %d, batch_size= %d, learning_rate= %f, q= %f, "
            "pretrain_e= %d, gen_old=  %d, "
            % (args.num_epochs, args.num_constrains, args.ml, alpha, args.batch_size, args.lr, args.q,
                args.epochs_pretrain, args.gen_old))

    if args.lrs == True:
        f.write("decay_rate= %f, epochs_lr= %d, name= %s. " % (args.decay_rate, args.epochs_lr, ex_name))
    else:
        f.write("decay= %f, name= %s. " % (args.decay, ex_name))

    f.write("Pretrain accuracy: %f , " % (pretrain_acc))
    f.write("Accuracy train: %f, NMI: %f, ARI: %f, sc: %f.\n" % (acc, nmi, ari, sc))

    rec, z_sample, p_z_c, p_c_z = model.predict([x_test, np.zeros(len(x_test))])
    yy = np.argmax(p_c_z, axis=-1)
    acc = utils.cluster_acc(y_test, yy)
    nmi = normalized_mutual_info_score(y_test, yy)
    ari = adjusted_rand_score(y_test, yy)

    acc_tot.append(acc)
    nmi_tot.append(nmi)
    ari_tot.append(ari)

    f.write("Accuracy test: %f, NMI: %f, ARI: %f.\n" % (acc, nmi, ari))
    f.close()
    print(str(acc))
    print(str(nmi))
    print(str(ari))

    # Save confusion matrix
    conf_mat = utils.make_confusion_matrix(y_test, yy, configs['training']['num_clusters'])
    np.save("logs/" + ex_name + "/conf_mat.npy", conf_mat)

    # Save embeddings
    if args.save_embedding:
        proj = ProjectorPlugin("logs/" + ex_name, z_sample)


        proj.save_labels(y_test)

        # Add images to projector

        if args.data == 'MNIST':
            proj.save_image_sprites(x_test, 28, 28, 1, True)

        proj.finalize()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to a config file.")

    _args = parser.parse_args()

    cfg = utils.load_config(_args)

    run_experiment(cfg)