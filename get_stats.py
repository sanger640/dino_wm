from datasets.jenga_dset import JengaDataset
dset = JengaDataset(data_path="/mnt/ssd_data/tasks/jenga_mujoco_noise", n_rollout=None)
print("Mean:", dset.action_mean)
print("Std:", dset.action_std)
print("Mean:", dset.proprio_mean)
print("Std:", dset.proprio_std)