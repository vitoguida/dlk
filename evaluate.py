from sklearn.linear_model import LogisticRegression
from utils import get_parser, load_all_generations, CCS
from sklearn.model_selection import train_test_split

def main(args, generation_args):
    # load hidden states and labels
    neg_hs, pos_hs, y = load_all_generations(generation_args)

    # Make sure the shape is correct
    assert neg_hs.shape == pos_hs.shape
    neg_hs, pos_hs = neg_hs[..., -1], pos_hs[..., -1]  # take the last layer
    if neg_hs.shape[1] == 1:  # T5 may have an extra dimension; if so, get rid of it
        neg_hs = neg_hs.squeeze(1)
        pos_hs = pos_hs.squeeze(1)

    # Very simple train/test split (using the fact that the data is already shuffled)
    """neg_hs_train, neg_hs_test = neg_hs[:int(len(neg_hs) * 0.7)], neg_hs[int(len(neg_hs) * 0.7):]
    pos_hs_train, pos_hs_test = pos_hs[:int(len(pos_hs) * 0.7)], pos_hs[int(len(pos_hs) * 0.7):]
    y_train, y_test = y[:int(len(y) * 0.7)], y[int(len(y) * 0.7):]"""

    neg_hs_train, neg_hs_test = train_test_split(neg_hs, test_size=0.4, random_state=42)
    pos_hs_train, pos_hs_test = train_test_split(pos_hs, test_size=0.4, random_state=42)

    y_train, y_test = train_test_split(y, test_size=0.4, random_state=42)

    # Make sure logistic regression accuracy is reasonable; otherwise our method won't have much of a chance of working
    # you can also concatenate, but this works fine and is more comparable to CCS inputs
    x_train = neg_hs_train - pos_hs_train  
    x_test = neg_hs_test - pos_hs_test
    lr = LogisticRegression(class_weight="balanced")
    lr.fit(x_train, y_train)
    print("Logistic regression accuracy: {}".format(lr.score(x_test, y_test)))

    # Set up CCS. Note that you can usually just use the default args by simply doing ccs = CCS(neg_hs, pos_hs, y)
    ccs = CCS(neg_hs_train, pos_hs_train, nepochs=args.nepochs, ntries=args.ntries, lr=args.lr, batch_size=args.ccs_batch_size, 
                    verbose=args.verbose, device=args.ccs_device, linear=args.linear, weight_decay=args.weight_decay, 
                    var_normalize=args.var_normalize)
    
    # train and evaluate CCS
    ccs.repeated_train()
    ccs_acc = ccs.get_acc(neg_hs_test, pos_hs_test, y_test)
    print("CCS accuracy: {}".format(ccs_acc))

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    # Calcola le differenze
    x = neg_hs - pos_hs  # shape: (N, D)

    # Applica PCA per ridurre a 2 dimensioni
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x)

    # Visualizza
    plt.figure(figsize=(8, 6))
    plt.scatter(x_pca[y == 0, 0], x_pca[y == 0, 1], c='red', label='Classe 0', alpha=0.6)
    plt.scatter(x_pca[y == 1, 0], x_pca[y == 1, 1], c='blue', label='Classe 1', alpha=0.6)
    plt.title('PCA delle differenze neg_hs - pos_hs')
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = get_parser()
    generation_args = parser.parse_args()  # we'll use this to load the correct hidden states + labels
    # We'll also add some additional args for evaluation
    parser.add_argument("--nepochs", type=int, default=1000)
    parser.add_argument("--ntries", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ccs_batch_size", type=int, default=-1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--ccs_device", type=str, default="cuda")
    parser.add_argument("--linear", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--var_normalize", action="store_true")
    args = parser.parse_args()
    main(args, generation_args)
