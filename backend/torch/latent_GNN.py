import torch
import sys
sys.path.append("")

from backend.torch.networks import LatentGNNV1

if __name__ == "__main__":
    ctxt_dim = 1 + 3 + 4*2
    latent_nodes = 6
    num_kernels = 2
    hidden_dict = dict(
    in_ = [64, 64],
    out_ = [32, 32]
    )
    network = LatentGNNV1(input_dim=ctxt_dim, hidden_dict = hidden_dict,
                        latent_dims=[latent_nodes, latent_nodes],
                        num_kernels=num_kernels,
                        output_size=4*2)
    
    dump_inputs = torch.rand((100, ctxt_dim))
    print(str(network)) 
    output = network(dump_inputs)
    print(output.shape)
#     test_group_latentgnn()