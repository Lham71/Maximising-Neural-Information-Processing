'''
Estimation/maximization of empowerment using Variational bound for linear system 
author: Elham 
The following code performs joint optimization: 1st: calculation of the empowerment 
via optimizaing variational lower bound and our proposed method of increasing the state horizon
2nd: maximization of the empowerment using parametrization of the system

'''

import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from matplotlib import pyplot as plt
from matplotlib import cm as CM
import datetime
import pickle

tfd = tfp.distributions
layers = tf.contrib.layers

tf.reset_default_graph()
tf.logging.set_verbosity(tf.logging.WARN)

#hyper_parameters
learning_rate_nn = 0.0001
epochs = 2500
sample_size = 1024 * 16
hidden_units = 12

sqr = np.sqrt(sample_size)
dim = 2
delta_t = 0.1
MI_sample = 100
plot_shape = 256

r1 = -1.0
r2 = 1.0

noise_power = 0.1
clip_val = 0.5


with tf.variable_scope('initialization'):
    '''
    Initializing the state space 
    '''

    XM = tf.placeholder(tf.float32, shape=(None, dim), name='XM')
    A = tf.constant([[-0.5, 0.], [0., -0.5]], name="A")
    #X_init = tf.random_uniform([sample_size, dim], minval=r1, maxval=r2, seed=0, name='X_init')
    x1, x2 = tf.meshgrid(tf.linspace(r1, r2, np.sqrt(sample_size)), tf.linspace(r2, r1, np.sqrt(sample_size)))
    xx1, xx2 = tf.reshape(x1, [-1]), tf.reshape(x2, [-1])
    X_init = tf.stack([xx1, xx2], 1, name="X_init")

xp1, xp2 = tf.meshgrid(tf.linspace(r1, r2, np.sqrt(plot_shape)), tf.linspace(r2, r1, np.sqrt(plot_shape)))
xxp1, xxp2 = tf.reshape(xp1, [-1]), tf.reshape(xp2, [-1])
X_plot = tf.stack([xxp1, xxp2], 1, name="X_plot")

with tf.variable_scope('nn_w'):
    
    '''
    building NN to model source (or potential actions) distribution
    '''

    a1_w = layers.fully_connected(inputs=XM, num_outputs=hidden_units, activation_fn=tf.nn.elu,
                                  weights_initializer=layers.xavier_initializer(seed=1), scope='layer1_w')
    tf.summary.histogram('a1_w', a1_w)

    a2_w = layers.fully_connected(inputs=a1_w, num_outputs=hidden_units, activation_fn=tf.nn.elu,
                                  weights_initializer=layers.xavier_initializer(seed=11), scope='layer2_w')

    tf.summary.histogram('a2_w', a2_w)

    # a3_w = layers.fully_connected(inputs=a2_w, num_outputs=hidden_units, activation_fn=tf.nn.elu,
    #                               weights_initializer=layers.xavier_initializer(seed=115), scope='layer3_w')
    # tf.summary.histogram('a3_w', a3_w)

    mu_w = layers.fully_connected(inputs=a2_w, num_outputs=dim, activation_fn=None,
                                  weights_initializer=layers.xavier_initializer(seed=15), scope='outMu_w')
    log_sigma_w = layers.fully_connected(inputs=a2_w, num_outputs=dim, activation_fn=None,
                                         weights_initializer=layers.xavier_initializer(seed=106), scope='outSig_w')
    sigma_w = tf.exp(log_sigma_w)


with tf.variable_scope('u_repar'):
    
    '''
    repapremtrization trick
    '''

    eps_n = tf.random_normal(shape=tf.shape(sigma_w), mean=0, stddev=1, seed=0, dtype=tf.float32)
    u = (mu_w + sigma_w * eps_n)
    u_clip = tf.clip_by_value(u, -clip_val, clip_val, name='input')

with tf.variable_scope('next_state'):
    '''
    evolving the state space through time for arbitrary number of time
    '''
    coeff = (tf.transpose(A) * delta_t) + tf.eye(dim)

    noise = tf.random_normal(shape=tf.shape(sigma_w), mean=0, stddev=1, seed=4, dtype=tf.float32)

    X_delta_t = tf.matmul(XM, coeff) + (u_clip * delta_t)
    #X_delta_t = tf.clip_by_value(X_delta_t, r1, r2, name='next_state_cliped')

    X_delta_t_2 = tf.matmul(X_delta_t, coeff)+ (noise_power * noise)
    X_delta_t_2 = tf.clip_by_value(X_delta_t_2, r1, r2, name='2step_next_state')

    # X_delta_t_3 = tf.matmul(X_delta_t_2, coeff)
    # #X_delta_t_3 = tf.clip_by_value(X_delta_t_3, r1, r2, name='3step_next_state')
    #
    # X_delta_t_4 = tf.matmul(X_delta_t_3, coeff)
    # #X_delta_t_4 = tf.clip_by_value(X_delta_t_4, r1, r2, name='4step_next_state')
    #
    # X_delta_t_5 = tf.matmul(X_delta_t_4, coeff) + (noise_power * noise)
    # X_delta_t_5 = tf.clip_by_value(X_delta_t_5, r1, r2, name='4step_next_state')

    # ===============================

    # X_delta_t = tf.matmul(XM, coeff) + tf.matmul(u_clip, coeff) + (0.25 * noise)
    # X_delta_t = tf.clip_by_value(X_delta_t, r1, r2, name='next_state_cliped')

with tf.variable_scope('vector_field'):

    vector_field = tf.matmul(XM, tf.transpose(A), name='vec_field')

with tf.variable_scope('nn_q', reuse=tf.AUTO_REUSE):
    
    '''
    building NN to approximate variational distribution
    '''

    input_concat = tf.concat([XM, X_delta_t_2], 1)

    a1_q = layers.fully_connected(inputs=input_concat, num_outputs=hidden_units, activation_fn=tf.nn.elu,
                                  weights_initializer=layers.xavier_initializer(seed=12), scope='layer1_q')
    tf.summary.histogram('a1_q', a1_q)

    a2_q = layers.fully_connected(inputs=a1_q, num_outputs=hidden_units, activation_fn=tf.nn.elu,
                                  weights_initializer=layers.xavier_initializer(seed=13), scope='layer2_q')
    tf.summary.histogram('a2_q', a2_q)

    # a3_q = layers.fully_connected(inputs=a2_q, num_outputs=hidden_units, activation_fn=tf.nn.elu,
    #                               weights_initializer=layers.xavier_initializer(seed=15), scope='layer3_q')
    # tf.summary.histogram('3_q', a3_q)

    mu_q = layers.fully_connected(inputs=a2_q, num_outputs=dim, activation_fn=None,
                                  weights_initializer=layers.xavier_initializer(seed=17), scope='outMu_q')

    log_sigma_q = layers.fully_connected(inputs=a2_q, num_outputs=dim, activation_fn=None,
                                         weights_initializer=layers.xavier_initializer(seed=108), scope='outSig_q')
    sigma_q = tf.exp(log_sigma_q)

with tf.variable_scope('Creating_cost'):

    mvn_w = tfd.MultivariateNormalDiag(loc=mu_w, scale_diag=sigma_w, name='mvn_w')
    mvn_q = tfd.MultivariateNormalDiag(loc=mu_q, scale_diag=sigma_q, name='mvn_q')

    ln_w = mvn_w.log_prob(u, 'LnW')
    ln_q = mvn_q.log_prob(u, 'LnQ')

    reward = tf.reduce_mean(ln_q - ln_w, name='reward')


    cost = -reward
    tf.summary.scalar('cost', reward)

t_vars = tf.trainable_variables()
NN_vars = t_vars

with tf.variable_scope('Model_Optimizer'):

    optimizer_nn = tf.contrib.opt.LazyAdamOptimizer(learning_rate=learning_rate_nn)

    grads_vars_nn = optimizer_nn.compute_gradients(cost, var_list=NN_vars)

    train_NN = optimizer_nn.minimize(cost, var_list=NN_vars)

    for grad, var in grads_vars_nn:
        tf.summary.histogram(var.name + '/gradient', grad)
        tf.summary.histogram(var.name + '/NN_variables', var)

with tf.variable_scope('source_planning'):
    tf.summary.histogram("mu_w1", mu_w[:, 0])
    tf.summary.histogram("mu_w2", mu_w[:, 1])
    tf.summary.histogram("mu_q1", mu_q[:, 0])
    tf.summary.histogram("mu_q2", mu_q[:, 1])
    tf.summary.histogram("sig_w1", sigma_w[:, 0])
    tf.summary.histogram("sig_w2", sigma_w[:, 1])
    tf.summary.histogram("sig_q1", sigma_q[:, 0])
    tf.summary.histogram("sig_q2", sigma_q[:, 1])


def variational_mutual_information(mu_s, sigma_s, mu_p, sigma_p, dimension, mi_sample, sample_num):

    mvn_n = tfd.MultivariateNormalDiag(loc=tf.zeros([sample_num, dimension]), scale_diag=tf.ones([sample_num, dimension]), name='mvn_N')
    eps_m = mvn_n.sample(sample_shape=mi_sample, seed=5, name='noiseI')

    u_w = mu_s + sigma_s * eps_m
    u_q = mu_p + sigma_p * eps_m

    mvn_s = tfd.MultivariateNormalDiag(loc=mu_s, scale_diag=sigma_s, name='mvn_wI')
    mvn_p = tfd.MultivariateNormalDiag(loc=mu_p, scale_diag=sigma_p, name='mvn_qI')

    ln_s = mvn_s.log_prob(u_w)
    ln_p = mvn_p.log_prob(u_q)
    MI = tf.reduce_mean(ln_p - ln_s, axis=0)

    return MI


def variational_mutual_inf_reg(mu_s, sigma_s, mu_p, sigma_p, dimension, mi_sample, sample_num):

    mvn_n = tfd.MultivariateNormalDiag(loc=tf.zeros([sample_num, dimension]), scale_diag=tf.ones([sample_num, dimension]), name='mvn_N')
    eps_m = mvn_n.sample(sample_shape=mi_sample, seed=5, name='noiseI')

    u_w = mu_s + sigma_s * eps_m

    mvn_s = tfd.MultivariateNormalDiag(loc=mu_s, scale_diag=sigma_s, name='mvn_wI')
    mvn_p = tfd.MultivariateNormalDiag(loc=mu_p, scale_diag=sigma_p, name='mvn_qI')

    ln_s = mvn_s.log_prob(u_w)
    ln_p = mvn_p.log_prob(u_w)
    MI = tf.reduce_mean(ln_p - ln_s, axis=0)

    return MI



MI_bound = variational_mutual_information(mu_w, sigma_w, mu_q, sigma_q, dim, MI_sample, sample_size)

MI_reg_bound = variational_mutual_inf_reg(mu_w, sigma_w, mu_q, sigma_q, dim, MI_sample, sample_size)

merged_summary_op = tf.summary.merge_all()



with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    X_init_, X_plot_ = sess.run([X_init, X_plot])


    #filename = "./liiin/run"
    #sum_writer = tf.summary.FileWriter(filename, graph=tf.get_default_graph())

    initial_MI, initial_reg_MI = sess.run([MI_bound, MI_reg_bound], feed_dict={XM: X_init_})

    X_current = X_init_
    #print('xinit', X_current)

    reward_nn = []

    for i in range(epochs):

        cost_, grads_, _, mu_w_, sigma_w_, mu_q_, sigma_q_, summary = \
            sess.run([cost, grads_vars_nn, train_NN, mu_w, sigma_w, mu_q, sigma_q, merged_summary_op], feed_dict={XM: X_current})

        s = np.arange(X_current.shape[0])
        np.random.seed(seed=i)

        np.random.shuffle(s)
        X_current = X_init_[s]
        #print('s', s)

        reward_nn.append(-cost_)
        #sum_writer.add_summary(summary, i)


    learned_MI, learned_MI_reg = sess.run([MI_bound, MI_reg_bound], feed_dict={XM: X_init_})
    #MI0 = sess.run(MI_bound, feed_dict={XM: np.zeros([1, 2])})
    vec_field_ = sess.run(vector_field, feed_dict={XM: X_plot_})
    # x_ev = np.array([[-1, -1.], [-0.5, -0.5], [0, 0], [0.5, 0.5], [1, 1], [-0.5, 0.5], [-1, 1.]])
    # mu, sigma, muq, sigmaq = sess.run([mu_w, sigma_w, mu_q, sigma_q], feed_dict={XM: x_ev})
    # print('x', x_ev)
    # print('## mu_w', mu), print('### sigma_w', sigma), print('## mu_q', muq), print('## sigma_q', sigmaq)


fig = plt.figure(figsize=(15, 8))

plt.subplot(231)
plt.plot(reward_nn)
plt.grid()

plt.subplot(232)
vmin = np.min(learned_MI)
vmax = np.max(learned_MI)
v = np.linspace(vmin, vmax, 7, endpoint=True)
plt.hexbin(X_init_[:, 0], X_init_[:, 1], C=learned_MI, gridsize=40, cmap=CM.jet, bins=None, vmin=vmin, vmax=vmax)
plt.title('learned MI bound')
plt.colorbar(ticks=v)
# plt.colorbar(ticks=v)
# plt.gca().tick_params(axis='both', which='major', labelsize=14)

plt.subplot(233)
dif = initial_MI
v1, v2 = np.min(dif), np.max(dif)
vdif = np.linspace(v1, v2, 7, endpoint=True)
plt.hexbin(X_init_[:, 0], X_init_[:, 1], C=dif, gridsize=40, cmap=CM.jet, bins=None, vmin=v1, vmax=v2)
plt.title('initial MI')
plt.colorbar(ticks=vdif)

plt.subplot(234)
plt.quiver(X_plot_[:, 0], X_plot_[:, 1], vec_field_[:, 0], vec_field_[:, 1], scale=30.)
plt.grid()

plt.subplot(235)
vm, vM = np.min(learned_MI_reg), np.max(learned_MI_reg)
v_reg = np.linspace(vm, vM, 7, endpoint=True)
plt.hexbin(X_init_[:, 0], X_init_[:, 1], C=learned_MI_reg,  gridsize=40, cmap=CM.jet, bins=None, vmin=vm, vmax=vM)
plt.title('MI reg')
plt.colorbar(ticks=v_reg)

plt.subplot(236)
dif_reg = learned_MI_reg - learned_MI
vd, vD = np.min(dif_reg), np.max(dif_reg)
v_reg_dif = np.linspace(vd, vD, 7, endpoint=True)
plt.hexbin(X_init_[:, 0], X_init_[:, 1], C=dif_reg, gridsize=40, cmap=CM.jet, bins=None, vmin=vd, vmax=vD)
plt.title('-reg ')
plt.colorbar(ticks=v_reg_dif)


plt.show()

















