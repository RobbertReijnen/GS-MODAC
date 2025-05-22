## GS-MODAC: Graph-Supported Multi-Objective Dynamic Algorithm Configuration

GS-MODAC introduces a novel methodology that combines Graph Neural Networks (GNNs) with Deep Reinforcement Learning (DRL) to dynamically configure Evolutionary Algorithms (EAs) for solving Multi-Objective Combinatorial Optimization (MOCO) problems. This approach captures the search process state as a graph in the objective space, enabling the DRL agent to dynamically tune EA parameters based on the current state of the search.


## GS-MODAC Framework

At each generation of the evolutionary algorithm, the population of candidate solutions is transformed into a graph. Nodes represent normalized objective values, and edges link solutions based on Pareto fronts. A GNN processes this graph to generate an embedding representing the search state. The DRL agent then outputs new EA parameter configurations, which guide the next iteration. This loop continues until the optimization terminates.

<img src="https://github.com/user-attachments/assets/2e67290e-8173-4dcd-bf56-c2c5bbe137cc" alt="GS-MODAC" style="max-width:50%; max-height:50%;">

---

## Citation:

If you use GS-MODAC in your research or work, please cite the following paper (Note: this citation will be updated upon publication):

```
@article{reijnengraph,
  title={Graph-Supported Dynamic Algorithm Configuration for Multi-Objective Combinatorial Optimization},
  author={Reijnen, Robbert and Wu, Yaoxin and Bukhsh, Zaharah and Zhang, Yingqian}
}
```
