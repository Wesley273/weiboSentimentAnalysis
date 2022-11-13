import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model",type=str,default="bigru",choices=["bigru","bilstm_attention"])
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--drop_prob", type=float, default=0.5)
parser.add_argument("--hidden_size",type=int,default=128)
parser.add_argument("--extra_embedding",type=bool,default=True)
parser.add_argument("--embedding_dim",type=int,default=100)

args = parser.parse_args()

if __name__ == "__main__":
    print(args.dataset)