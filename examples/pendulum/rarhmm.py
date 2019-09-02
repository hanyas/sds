import autograd.numpy as np
import autograd.numpy.random as npr

from sds import rARHMM
from sds.utils import sample_env


if __name__ == "__main__":

    np.random.seed(1337)

    import matplotlib.pyplot as plt

    from hips.plotting.colormaps import gradient_cmap
    import seaborn as sns

    sns.set_style("white")
    sns.set_context("talk")

    color_names = ["windows blue", "red", "amber", "faded green", "dusty purple", "orange"]

    colors = sns.xkcd_palette(color_names)
    cmap = gradient_cmap(colors)

    import pickle
    import gym
    import rl

    env = gym.make('Pendulum-RL-v0')
    env._max_episode_steps = 5000
    env.seed(1337)

    nb_rollouts, nb_steps = 20, 200
    dm_obs = env.observation_space.shape[0]
    dm_act = env.action_space.shape[0]

    reps_ctl = pickle.load(open("reps_pendulum_ctl.pkl", "rb"))
    obs, act = sample_env(env, nb_rollouts, nb_steps)

    nb_states = 5

    rarhmm = rARHMM(nb_states, dm_obs, dm_act, type='recurrent')
    rarhmm.initialize(obs, act)
    lls = rarhmm.em(obs, act, nb_iter=50, prec=0., verbose=True)

    # plt.figure(figsize=(5, 5))
    # plt.plot(lls)
    # plt.show()
    #
    # plt.figure(figsize=(8, 8))
    # _idx = npr.choice(nb_rollouts)
    # _, _sample_obs = rarhmm.sample([act[_idx]], horizon=[nb_steps])
    # plt.plot(_sample_obs[0])
    # plt.show()
    #
    # _seq = npr.choice(len(obs))
    # _, z = rarhmm.viterbi(obs[_seq], act[_seq])
    #
    # x = rarhmm.mean_observation(obs, act)
    #
    # plt.figure(figsize=(8, 8))
    # plt.subplot(211)
    # plt.imshow(z[0][None, :], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors) - 1)
    # plt.xlim(0, len(z[0]))
    # plt.ylabel("$state_{\\mathrm{true}}$")
    # plt.yticks([])
    #
    # plt.subplot(212)
    # plt.plot(x[_seq], '-k', lw=2)
    # plt.xlim(0, len(x[_seq]))
    # plt.ylabel("$obs_{\\mathrm{inferred}}$")
    # plt.xlabel("time")
    #
    # plt.tight_layout()
    # plt.show()
    #
    # # testing
    # nb_rollouts, nb_steps = 5, 200
    # obs, act = sample_env(env, nb_rollouts, nb_steps)
    #
    # # error
    # mse, norm_mse = rarhmm.kstep_mse(obs, act, horizon=25,
    #                                  stoch=False, infer='viterbi')
    # print(mse, norm_mse)
    #
    # # forcasting
    # nb_rows = rarhmm.dm_obs
    # nb_cols = 5
    #
    # T, H = 50, 25
    # hist_obs, hist_act, nxt_act = [], [], []
    # for _obs, _act in zip(obs, act):
    #     hist_obs.append(_obs[:T, :])
    #     hist_act.append(_act[:T, :])
    #     nxt_act.append(_act[T: T + H, :])
    #
    # hr = [H for _ in hist_obs]
    # z_hat, obs_hat = rarhmm.forcast(hist_obs=hist_obs, hist_act=hist_act,
    #                                 nxt_act=nxt_act, horizon=hr, stoch=False)
    #
    # fig, axs = plt.subplots(nrows=nb_rows, ncols=nb_cols, figsize=(12, 4))
    # labels = ['x', 'y', 'v']
    # colors = ['r', 'g', 'b']
    # for i in range(nb_rows):
    #     for j in range(nb_cols):
    #         axs[i, j].plot(obs_hat[j][:, i], color=colors[i])
    #         axs[i, j].plot(obs[j][T: T + 1 + H, i], ':', color=colors[i])
    #         axs[i, j].set_xlabel('t')
    #         axs[i, j].set_ylabel(labels[i])
