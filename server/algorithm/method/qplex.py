# coding=utf-8
import tensorflow as tf
import numpy as np
import datetime
import time
import os
import random
import threading
import queue

from algorithm.method.network import Qplex_mixer


exp_name = 'DC-Ms'
class Method(object):
    """ For Qmix """
    def __init__(self, agent, num_global_s, num_s, num_a, name, test, lr=0.0001, gamma=0.99, replace_target_iter=1000,
                 memory_size=2000000, batch_size=256, epsilon=1, epsilon_decay=0.0001):
        self.agent = agent
        self.n_agents = 3
        self.name = name
        if exp_name.endswith('Sp'):
            self.num_global_s = num_global_s  # Sp
        elif exp_name.endswith('Ms'):
            self.num_global_s = 3 * num_s  # Ms
        else:
            assert exp_name.endswith('Oa')
            self.num_global_s = 3 * (num_s + num_a)  # Oa

        self.num_s = num_s
        self.num_a = num_a
        self.lr = lr
        self.gamma = gamma
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.test = test
        self.epsilon_min = 0.1
        self.learn_step_cnt = 0  # total learning step
        self.episode_cnt = 0
        self.memory = []
        self.memory_counter = 0
        self.load_latest_checkpoints = True
        self.exp_splicer = {}
        self.exp_splicing_lock = threading.RLock()
        self.update_queue = queue.Queue()
        self._build_net()
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name + '/target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name + '/eval_net')
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name + '/mixing_net/eval_hyper')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name + '/mixing_net/target_hyper')
        
        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()
        # if not self.test:
        self.train_writer = tf.summary.FileWriter('./data/logs-'+exp_name+'/' + self.name + '/' + 'dqn_lr_' + str(self.lr) + '_' +
                                                  datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'),
                                                  self.sess.graph)
        tf.summary.merge_all()
        self.cost_his = []
        self.saver = tf.train.Saver(max_to_keep=10000)

        if not self.test:
            self.sess.run(tf.global_variables_initializer())
            print('##### training mode: {} #####'.format(self.name))
        else:
            tf.reset_default_graph()
            path = './data/checkpoints/' + self.name + '/'
            if self.load_latest_checkpoints:
                checkpoint = tf.train.latest_checkpoint(path)
            else:
                model_list = {'33_attack': 360000, '33_defense': 450000, '33_freeball': 480000,
                              '33_ballclear': 320000, '33_assist': 390000}
                checkpoint = path + 'qmix.ckpt-' + str(model_list[self.name])
            self.saver.restore(self.sess, checkpoint)
            print('##### testing mode: {} | model:{} #####'.format(self.name, checkpoint))

        # start learning thread
        self.start_learning_thread()

    def _build_net(self):  # we use parameter sharing among agents
        with tf.variable_scope(self.name):
            # ------------------ all inputs ------------------------
            self.S = tf.placeholder(tf.float32, [None, self.num_global_s], name='S')  # input Global State
            self.s = tf.placeholder(tf.float32, [None, self.num_s], name='s1')  # input state for agent1
            self.S_ = tf.placeholder(tf.float32, [None, self.num_global_s], name='S_')  # input Next Global State
            self.s_ = tf.placeholder(tf.float32, [None, self.num_s], name='s1_')  # input next state for agent1
            self.R = tf.placeholder(tf.float32, [None, ], name='R')  # input Reward
            self.a = tf.placeholder(tf.float32, [None, self.num_a], name='a')  # input Action onehot for agent1
            self.a_ = tf.placeholder(tf.float32, [None, self.num_a], name='a')  # input Action onehot for agent1
            self.done = tf.placeholder(tf.float32, [None, ], name='done')  # input Done info ???

            self.q_m =  tf.placeholder(tf.float32, [None, ], name='q_value_max')
            self.q_m_ = tf.placeholder(tf.float32, [None, ], name='q_value_next_max')

            w_initializer, b_initializer = tf.random_normal_initializer(0., 0.1), tf.constant_initializer(0.0)

            # ------------------ build evaluate_net ------------------
            with tf.variable_scope('eval_net'):
                a_fc1 = tf.layers.dense(self.s, 128, tf.nn.relu, kernel_initializer=w_initializer,
                                        bias_initializer=b_initializer, name='agent_fc1_e')
                a_fc2 = tf.layers.dense(a_fc1, 128, tf.nn.relu, kernel_initializer=w_initializer,
                                        bias_initializer=b_initializer, name='agent_fc2_e')
                a_fc3 = tf.layers.dense(a_fc2, 64, tf.nn.relu, kernel_initializer=w_initializer,
                                        bias_initializer=b_initializer, name='agent_fc3_e')
                self.q_eval = tf.layers.dense(a_fc3, self.num_a, kernel_initializer=w_initializer,
                                              bias_initializer=b_initializer, name='q_e')

            # ------------------ build target_net ------------------
            with tf.variable_scope('target_net'):
                a_fc1_ = tf.layers.dense(self.s_, 128, tf.nn.relu, kernel_initializer=w_initializer,
                                         bias_initializer=b_initializer, name='agent_fc1_t')
                a_fc2_ = tf.layers.dense(a_fc1_, 128, tf.nn.relu, kernel_initializer=w_initializer,
                                         bias_initializer=b_initializer, name='agent_fc2_t')
                a_fc3_ = tf.layers.dense(a_fc2_, 64, tf.nn.relu, kernel_initializer=w_initializer,
                                         bias_initializer=b_initializer, name='agent_fc3_t')
                self.q_next = tf.layers.dense(a_fc3_, self.num_a, kernel_initializer=w_initializer,
                                              bias_initializer=b_initializer, name='q_t')

            # [batch*n_agents, 1]
            self.q_selected = tf.reduce_sum(tf.multiply(self.q_eval, self.a), axis=1)

            # ------------------ build mixing_net ------------------
            with tf.variable_scope('mixing_net'):
                # [batch, n_agents]
                self.q_concat = tf.reshape(self.q_selected, [-1, self.n_agents])
                self.q_concat_ =tf.reshape(self.q_m_, [-1, self.n_agents]) 

                with tf.variable_scope('eval_hyper'):
                    ans_chosen = Qplex_mixer(self.q_concat, self.S, self.n_agents, self.num_global_s, is_v=True)
                    ans_adv = Qplex_mixer(self.q_concat, self.S, self.n_agents, self.num_global_s, max_q_i=self.q_m, actions=self.a, is_v=False)
                    self.Q_tot = ans_chosen + ans_adv

                with tf.variable_scope('target_hyper'):
                    ans_chosen_ = Qplex_mixer(self.q_concat_, self.S_, self.n_agents, self.num_global_s, is_v=True)
                    ans_adv_ = Qplex_mixer(self.q_concat_, self.S_, self.n_agents, self.num_global_s, max_q_i=self.q_m_, actions=self.a_, is_v=False)
                    self.Q_tot_ = ans_chosen_ + ans_adv_

            # todo: add q_target, loss, train_op
            with tf.variable_scope('q_target'):
                q_target = self.R + (1 - self.done) * self.gamma * self.Q_tot_
                self.q_target = tf.stop_gradient(q_target)
            with tf.variable_scope('loss'):
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.Q_tot, name='TD_error'))
            with tf.variable_scope('train'):
                self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def act(self, state, avas, id, no_train):  # epsilon greedy
        if np.random.uniform() > self.epsilon or self.test or no_train:  # pick the argmax action
            s = np.array(state)
            if len(s.shape) < 2:
                s = np.array(state)[np.newaxis, :]
            q_eval = self.sess.run(self.q_eval, feed_dict={self.s: s})[0]
            q_eval[avas == 0] = -float('inf')  # unavailable actions should never be selected !
            action = np.argmax(q_eval)
        else:  # pick random action
            avail_action_dim = sum(avas)
            action = np.random.randint(0, avail_action_dim)
        return action

    def store(self, uuid, id, scene, exp):
        # we need to do exp splicing here to make exp of different game clients and interfaces together
        self.exp_splicing_lock.acquire()

        if uuid not in self.exp_splicer:
            self.exp_splicer[uuid] = [None, None, None]
        self.exp_splicer[uuid][id] = exp
        # print(self.exp_splicer)
        if None not in self.exp_splicer[uuid]:

            # time to splice the exps
            S1, s1, a1, r1, S1_, s1_, avas1, done1 = self.exp_splicer[uuid][0]
            S2, s2, a2, r2, S2_, s2_, avas2, done2 = self.exp_splicer[uuid][1]
            S3, s3, a3, r3, S3_, s3_, avas3, done3 = self.exp_splicer[uuid][2]
            R = r1 + r2 + r3
            done = all([done1, done2, done3])

            # reconstruct the experience
            if exp_name.endswith('Ms'):
                EXP1 = [s1+s2+s3, s1, s2, s3, a1, a2, a3, R, s1_+s2_+s3_, s1_, s2_, s3_, avas1, avas2, avas3, done]
                EXPs = [EXP1]

            ### Oa: add ongoing-action info to the global states (add a to S and S_)
            elif exp_name.endswith('Oa'):
                a1_onehot, a2_onehot, a3_onehot = [np.zeros(self.num_a) for _ in range(3)]
                a1_onehot[a1] = 1
                a2_onehot[a2] = 1
                a3_onehot[a3] = 1
                s1_g = np.concatenate((s1, a1_onehot))
                s1_g_ = np.concatenate((s1_, a1_onehot))
                s2_g = np.concatenate((s2, a2_onehot))
                s2_g_ = np.concatenate((s2_, a2_onehot))
                s3_g = np.concatenate((s3, a3_onehot))
                s3_g_ = np.concatenate((s3_, a3_onehot))
                EXP1 = [np.concatenate((s1_g, s2_g, s3_g)), s1, s2, s3, a1, a2, a3, R,
                        np.concatenate((s1_g_, s2_g_, s3_g_)), s1_, s2_, s3_, avas1, avas2, avas3, done]
                EXPs = [EXP1]

            ### Sp
            else:
                assert exp_name.endswith('Sp')
                EXP1 = [S1, s1, s2, s3, a1, a2, a3, R, S1_, s1_, s2_, s3_, avas1, avas2, avas3, done]
                EXP2 = [S2, s1, s2, s3, a1, a2, a3, R, S2_, s1_, s2_, s3_, avas1, avas2, avas3, done]
                EXP3 = [S3, s1, s2, s3, a1, a2, a3, R, S3_, s1_, s2_, s3_, avas1, avas2, avas3, done]
                EXPs = [EXP1, EXP2, EXP3]

            for EXP in EXPs:
                # if None not in exp:
                self.memory_counter += 1
                if len(self.memory) > self.memory_size:
                    # random replacement
                    index = np.random.randint(0, self.memory_size)
                    self.memory[index] = EXP
                else:
                    self.memory.append(EXP)
            self.exp_splicer[uuid] = [None, None, None]

            # check if it is ok to update
            if self.memory_counter % 5 == 0 and len(self.memory) > self.batch_size:
                self.update_queue.put(1)

        self.exp_splicing_lock.release()

    # the updating thread
    def start_learning_thread(self):
        if not self.test:
            print('start {} learning thread...'.format(self.name))
            learning_thread = threading.Thread(target=self.learn, name=self.name+'_learning_thread')
            learning_thread.setDaemon(True)
            learning_thread.start()
        else:
            print('testing...')

    def learn(self):
        while True:
            if self.update_queue.get():
                for _ in range(4):
                    self.update()

    def update(self):
        # sample batch exp from memory
        if self.learn_step_cnt % 10000 == 0:
            print(self.name, 'update ----> learn_step_cnt', self.learn_step_cnt)
        batch_exp = random.sample(self.memory, self.batch_size)
        S, s, a, R, S_, s_, avas, done = [[] for _ in range(8)]
        for exp in batch_exp:
            S.append(exp[0])
            s.append([exp[1], exp[2], exp[3]])
            a.append([exp[4], exp[5], exp[6]])
            R.append(exp[7])
            S_.append(exp[8])
            s_.append([exp[9], exp[10], exp[11]])
            avas.append([exp[12], exp[13], exp[14]])
            done.append(exp[15])
        # to get q_tot
        s = np.stack(s)
        a = np.stack(a)
        s_ = np.stack(s_)
        avas = np.stack(avas)
        s.shape = (self.batch_size*self.n_agents, self.num_s)
        s_.shape = (self.batch_size*self.n_agents, self.num_s)
        avas.shape = (self.batch_size*self.n_agents, self.num_a)

        actions_1hot = np.zeros([self.batch_size, self.n_agents, self.num_a], dtype=int)
        grid = np.indices((self.batch_size, self.n_agents))
        actions_1hot[grid[0], grid[1], a] = 1
        actions_1hot.shape = (self.batch_size*self.n_agents, self.num_a)


        q = self.sess.run(self.q_eval, feed_dict={self.s: s})
        q_m = np.max(q, axis=1)

        # to get q_tot_
        q_ = self.sess.run(self.q_next, feed_dict={self.s_: s_})
        q_[avas[:, :] == 0] = - 999999  # mask unavailable actions
        q_m_ = np.max(q_, axis=1)
        max_q_i = np.argmax(q_, axis=1)
        max_q_i.shape = (self.batch_size, self.n_agents)

        actions_1hot_ = np.zeros([self.batch_size, self.n_agents, self.num_a], dtype=int)
        grid_ = np.indices((self.batch_size, self.n_agents))
        actions_1hot_[grid_[0], grid_[1], max_q_i] = 1
        actions_1hot_.shape = (self.batch_size*self.n_agents, self.num_a)


        q_tot_ = self.sess.run(self.Q_tot_, feed_dict={self.S_: S_, self.q_m_: q_m_, self.a_: actions_1hot_})

        # update
        _, cost = self.sess.run([self._train_op, self.loss],
                                feed_dict={self.S: S, self.s:s, self.a: actions_1hot,
                                           self.R: R, self.Q_tot_: q_tot_, self.done: done, self.q_m: q_m})
        # print('cost', cost)

        self.write_summary_scalar('loss', cost, self.learn_step_cnt)
        self.write_summary_scalar('epsilon', self.epsilon, self.learn_step_cnt)
        self.write_summary_scalar('memory_cnt', self.memory_counter, self.learn_step_cnt)
        self.save()  # save model
        self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)  # decay epsilon
        self.learn_step_cnt += 1

        # check to do the soft replacement of target net
        if self.learn_step_cnt % self.replace_target_iter == 0 and self.learn_step_cnt:
            self.sess.run(self.target_replace_op)

    def save(self):
        if self.learn_step_cnt % 10000 == 0 and self.learn_step_cnt > 0 and not self.test:
            path = './data/checkpoints/' + self.name + '/'
            if not os.path.exists(path): os.makedirs(path)
            model_name = os.path.join(path, 'qplex.ckpt')
            save_path = self.saver.save(self.sess, model_name, global_step=self.learn_step_cnt)
            # print('save model %s' % save_path)

    def episode_done(self):
        self.episode_cnt += 1
        # self.config.eps = max(self.config.eps - self.config.eps_dec, self.config.eps_min)

    def write_summary_scalar(self, tag, value, iteration):
        self.train_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)]), iteration)





