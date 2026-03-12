import lmdb
import pickle

# Update this path if your LMDB is located somewhere else
lmdb_path = "/home/sanger/jenga_mujoco_noise/jenga_unified.lmdb"

def print_lmdb_stats():
    print(f"Opening LMDB at: {lmdb_path}")
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    
    with env.begin(write=False) as txn:
        meta_bytes = txn.get(b"__metadata__")
        
        if meta_bytes is None:
            print("❌ Error: Could not find '__metadata__' key in the LMDB.")
            return
            
        metadata = pickle.loads(meta_bytes)
        stats = metadata.get("stats", {})
        
        print("\n--- Normalization Statistics ---")
        print(f"Action Mean:  {stats.get('action_mean')}")
        print(f"Action Std:   {stats.get('action_std')}")
        print(f"Proprio Mean: {stats.get('proprio_mean')}")
        print(f"Proprio Std:  {stats.get('proprio_std')}")
        print("--------------------------------\n")
        
    env.close()

if __name__ == "__main__":
    print_lmdb_stats()