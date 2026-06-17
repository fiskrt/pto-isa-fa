The goal is to have a flash attention 2 like implementation in pto-isa. See 3p/pto-isa/ for which NPU ops are available.

The sourounding scaffolding should be miminal and simple, for example a short jit method, short set up and run verification benchmarks etc, just in a few LoC.

The flash attention 2 implementation in pto-isa should:

1. Be divided into 4 clean steps where we calculate QKt
2. 


where the goal is to have it run as fast as matrix multiplication, i.e. we must make sure that the online softmax runs in parallel while the cube cores are working on matrix multiplication.

Another very import aspect is load balancing:

And here we will have a few cases. The most relevant setting is the `causal=True` as it's harder to balance
since each row loops over a different amount of tiles due to the mask. 

### Case 1: num tiles in S0 dim is a multiple of 2*C where C=24 (num cores)

Let's say we have a S0xS1 matrix, with tile dims SO_tile x S1_tile, and S0/S0_tile=n_0 and S1/S1_tile=n_1. The assumption here is that 2*C divides n_0. Then we assign core i to handle rows from bottom and top, such that each row contains n_1+1 tiles. Hence perfectly load balances the rows.

But this case might only be relevent for longer seqences as S0_tile is going to be between 32-256 for optimal cube core utilization.

To extend this loadbalancing to cases where it's not a perfect multiple, we can do the same assigning bottom and top rows and the just do round robin for the middle rows that are left over. This is not so bad since the rows close to the middle will be more similar in length than row 0 and row n-1 for example.


### case 2: decoding so n_0=1 or S0=1

Might have to do load balancing in the S1 dimension aswell, but this requires keeping track of LSE and some more work. But this is for future work.