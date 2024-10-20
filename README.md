# Buffalo: Enabling Large-Scale GNN Training via Memory-Efficient Bucketization  



## install requirements:
 The framework of Buffalo is developed upon DGL(pytorch backend)  
 We use Ubuntu 20.04, CUDA 12.3 (it's also compatible with Ubuntu18.04, CUDA 11.2, the package version you need to install are denoted in install_requirements.sh).  
 The requirements:  pytorch >= 1.9, DGL >= 0.8

`bash install_requirements.sh`.  

## Our main contributions: 
Buffalo introduces a system addressing the bucket explosion and enabling load balancing between graph partitions for GNN training. 




Buffalo provides bucket-level partitioning and scheduling algorithm.   
 
The overall time complexity of Buffalo’s algorithm (algorithm 3 in the paper)can be summarized as follows:  

### Overall Complexity  
The algorithm's time complexity is:  
**$$O(D + K_{max} \cdot (S + G + M))$$**  

### Components:  
- **$$D$$**: Time for degree bucketing, calculated as **$$O(V + E)$$** (where $$V$$ is nodes and $$E$$ is edges).  
- **$$K_{max}$$**: Maximum number of partitions (micro-batches).  
- **$$S$$**: Time for splitting buckets, **$$O(b)$$**. $$b$$ is the number of output nodes in the bucket to be split.  
- **$$G$$**: Time for balancing memory, calculated as **$$O(n \cdot W)$$** (where $$n$$ is buckets and $$W$$ is memory capacity).  
- **$$M$$**: Time for generating micro-batches, which includes:  
  -  **Parallel Processing**: Can reduce time to **$$O(d)$$** if operations are parallelized. $$d$$ is the degree of center nodes.  



 
