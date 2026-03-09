import lmdb

# Point this to your original scratch directory
lmdb_path = "/home/ali313/scratch/jenga_mujoco_noise/jenga_images.lmdb"

env = lmdb.open(lmdb_path, readonly=True, lock=False)
with env.begin() as txn:
    cursor = txn.cursor()
    print("🔍 --- First 15 Keys in LMDB ---")
    for i, (key, _) in enumerate(cursor):
        print(key.decode('ascii'))
        if i >= 14:
            break
env.close()