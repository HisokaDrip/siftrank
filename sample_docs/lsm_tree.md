# LSM tree
Writes go to an in-memory memtable and a write-ahead log for crash recovery. When the memtable fills it is flushed to a sorted SSTable on disk. Each SSTable has a bloom filter so lookups can skip tables that definitely don't contain the key. Compaction merges tables and drops tombstones.
