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



def pretrain(cfg, model, performance_logger):
    inp_shape = cfg['model']['inp_shape']

    # Get the AE from DCGMM
    input = tfkl.Input(shape=inp_shape)

    if cfg['model']['type'] == "FC":
        f = tfkl.Flatten()(input)
        e1 = model.encoder.dense1(f)
        e2 = model.encoder.dense2(e1)
        e3 = model.encoder.dense3(e2)
        z = model.encoder.mu(e3)
        d1 = model.decoder.dense1(z)
        d2 = model.decoder.dense2(d1)
        d3 = model.decoder.dense3(d2)
        dec = model.decoder.dense4(d3)
    elif cfg['model']['type'] == "CNN":
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
    elif cfg['model']['type'] == "VGG":
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

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    if cfg['dataset']['name'] == 'MNIST':
        autoencoder.compile(optimizer=optimizer, loss="binary_crossentropy")
    else:
        autoencoder.compile(optimizer=optimizer, loss="mse")
    
    x_train, x_test, y_train, y_test = utils.get_data(cfg)

    X = np.concatenate((x_train, x_test))
    Y = np.concatenate((y_train, y_test))


    pretrain_path = os.path.join(cfg['dir']['pretrain'],cfg['dataset']['name'],'autoencoder','cp.ckpt')
    gmm_save_path = os.path.join(cfg['dir']['pretrain'],cfg['dataset']['name'],'gmm_save.sav')

    # If the model should be run from scratch:
    if cfg['experiment']['pretrain']:
        performance_logger.info('started pretraining')
        print('started pretraining')

        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=pretrain_path,
                                                         save_weights_only=True, verbose=1)
        autoencoder.fit(X, X, epochs=cfg['experiment']['epochs_pretrain'], batch_size=32, callbacks=cp_callback)

        encoder = model.encoder
        input = tfkl.Input(shape=inp_shape)
        z, _ = encoder(input)
        z_model = tf.keras.models.Model(inputs=input, outputs=z)
        z = z_model.predict(X)

        estimator = GaussianMixture(n_components=cfg['training']['num_clusters'], covariance_type='diag', n_init=3)
        estimator.fit(z)
        pickle.dump(estimator, open(gmm_save_path, 'wb'))

        performance_logger.info('finished pretraining')
        print('finished pretraining')

    else:
        autoencoder.load_weights(pretrain_path)
        estimator = pickle.load(open(gmm_save_path, 'rb'))

        performance_logger.info('loaded mnist pretrained weights')
        print('loaded mnist pretrained weights')

    
    encoder = model.encoder
    input = tfkl.Input(shape=inp_shape)
    z, _ = encoder(input)
    z_model = tf.keras.models.Model(inputs=input, outputs=z)

    # Assign weights to GMM mixtures of DCGMM
    mu_samples = estimator.means_
    sigma_samples = estimator.covariances_
    model.c_mu.assign(mu_samples)
    model.log_c_sigma.assign(np.log(sigma_samples))

    yy = estimator.predict(z_model.predict(X))
    pretrain_acc = utils.cluster_acc(yy, Y)

    performance_logger.info(f'pretrain accuracy: {pretrain_acc}')
    print(f'pretrain accuracy: {pretrain_acc}')

    return model


def run_experiment(cfg):
    
    if cfg['experiment']['name'] is None:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        ex_name = "{}_{}".format(str(timestr), uuid.uuid4().hex[:5])
    else:
        ex_name = cfg['experiment']['name']

    experiment_path = os.path.join(cfg['dir']['logging'], cfg['dataset']['name'], ex_name)
    Path(experiment_path).mkdir(parents=True)
    
    performance_logger = utils.get_logger(experiment_path, ex_name)

    performance_logger.info(f'cfg: {cfg}')
    performance_logger.info(f'num GPUs: {len(tf.config.list_physical_devices("GPU"))}')

    alpha = utils.get_alpha(cfg)

    x_train, x_test, y_train, y_test = utils.get_data(cfg)

    generator = DataGenerator(x_train, y_train, num_constrains=cfg['training']['num_constrains'], alpha=alpha, q=cfg['training']['q'],
                        batch_size=cfg['training']['batch_size'], ml=cfg['training']['ml'])
    train_generator = generator.gen()
    test_generator = DataGenerator(x_test, y_test, batch_size=cfg['training']['batch_size']).gen()

    model = utils.get_model(cfg)

    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg['training']['learning_rate'], decay=cfg['training']['decay'])


    # needs to be changed
    if cfg['training']['lrs']:
        cp_callback = [tf.keras.callbacks.TensorBoard(log_dir=experiment_path),
                        tf.keras.callbacks.LearningRateScheduler(utils.get_learning_rate_scheduler(cfg))]
        optimizer = tf.keras.optimizers.Adam(learning_rate=cfg['training']['learning_rate'])

    elif cfg['experiment']['save_model']:
        checkpoint_path = os.path.join(cfg['dir']['checkpoint'], cfg['dataset']['name'], ex_name)
        Path(checkpoint_path).mkdir(parents=True)

        cp_callback = [tf.keras.callbacks.TensorBoard(log_dir=experiment_path),
                        tf.keras.callbacks.ModelCheckpoint(
                            filepath=checkpoint_path,
                            verbose=1,
                            save_weights_only=True,
                            period=100)]
    else:
        cp_callback = [tf.keras.callbacks.TensorBoard(log_dir=experiment_path)]

    
    # pretrain model
    model = pretrain(cfg, model, performance_logger)

    # train model
    model.compile(optimizer, loss={"output_1": utils.get_loss_fn(cfg)}, metrics={"output_4": utils.accuracy_metric})

    model.fit(train_generator, validation_data=test_generator, steps_per_epoch=int(len(y_train)/cfg['training']['batch_size']), 
              validation_steps=len(y_test)//cfg['training']['batch_size'], epochs=cfg['training']['epochs'], callbacks=cp_callback)


    # measure training performance
    rec, z_sample, p_z_c, p_c_z = model.predict([x_train, np.zeros(len(x_train))])
    yy = np.argmax(p_c_z, axis=-1)
    acc = utils.cluster_acc(y_train, yy)
    nmi = normalized_mutual_info_score(y_train, yy)
    ari = adjusted_rand_score(y_train, yy)
    ml_ind1 = generator.ml_ind1
    ml_ind2 = generator.ml_ind2
    cl_ind1 = generator.cl_ind1
    cl_ind2 = generator.cl_ind2
    count = 0
    if cfg['training']['num_constrains'] == 0:
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

    performance_logger.info("Train Accuracy: %f, NMI: %f, ARI: %f, sc: %f." % (acc, nmi, ari, sc))


    # measure test performance
    rec, z_sample, p_z_c, p_c_z = model.predict([x_test, np.zeros(len(x_test))])
    yy = np.argmax(p_c_z, axis=-1)
    acc = utils.cluster_acc(y_test, yy)
    nmi = normalized_mutual_info_score(y_test, yy)
    ari = adjusted_rand_score(y_test, yy)

    performance_logger.info("Test Accuracy: %f, NMI: %f, ARI: %f.\n" % (acc, nmi, ari))

    # save test set confusion matrix
    conf_mat = utils.make_confusion_matrix(y_test, yy, cfg['model']['num_clusters'])
    np.save(os.path.join(experiment_path,'conf_mat.npy'), conf_mat)

    # save test set embeddings
    if cfg['experiment']['save_embedding']:
        proj = ProjectorPlugin(experiment_path, z_sample)

        proj.save_labels(y_test)

        # Add images to projector
        if cfg['dataset']['name'] == 'MNIST':
            proj.save_image_sprites(x_test, 28, 28, 1, True)

        proj.finalize()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to a config file.")

    _args = parser.parse_args()

    cfg = utils.load_config(_args)

    run_experiment(cfg)