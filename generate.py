from utils import get_parser, load_model, get_dataloader, get_all_hidden_states, save_generations,get_local_dataloader

def main(args):
    # Set up the model and data
    print("Loading model")
    model, tokenizer, model_type = load_model(args.model_name, args.cache_dir, args.parallelize, args.device)

    print("Loading dataloader")
    #dataloader = get_dataloader(args.dataset_name, args.split, tokenizer, args.prompt_idx, batch_size=args.batch_size,
                               # num_examples=args.num_examples, model_type=model_type, use_decoder=args.use_decoder, device=args.device)
    dataloader_local = get_local_dataloader("dataset/ml-1m/movies.dat",tokenizer, batch_size=args.batch_size,num_examples=args.num_examples, model_type=model_type, device=args.device)


    # Get the hidden states and labels
    print("Generating hidden states")
    neg_hs, pos_hs, y = get_all_hidden_states(model, dataloader_local, layer=args.layer, all_layers=args.all_layers,
                                              token_idx=args.token_idx, model_type=model_type, use_decoder=args.use_decoder)

    # Save the hidden states and labels
    print("Saving hidden states")
    save_generations(neg_hs, args, generation_type="negative_hidden_states")
    save_generations(pos_hs, args, generation_type="positive_hidden_states")
    save_generations(y, args, generation_type="labels")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)





"""#caricamento dataset personalizzato
    # Percorso del file
    file_path = "dataset/ml-1m/movies.dat"
    movies = pd.read_csv(file_path, sep="::", engine="python", names=["movie_id", "title", "genre"], encoding="latin-1")
    raw_dataset = Dataset.from_pandas(movies)"""




"""
class LocalContrastDataset(Dataset):
    def __init__(self, raw_dataset, tokenizer, model_type="encoder_decoder", use_decoder=False, device="cuda"):
        self.raw_dataset = raw_dataset
        self.tokenizer = tokenizer
        self.device = device
        self.model_type = model_type
        self.use_decoder = use_decoder
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.raw_dataset)

    def encode(self, question, answer):
        if self.model_type == "encoder_decoder":
            input_ids = self.tokenizer(question, answer, truncation=True, padding="max_length", return_tensors="pt")
            decoder_input_ids = self.tokenizer("", return_tensors="pt")
            input_ids["decoder_input_ids"] = decoder_input_ids["input_ids"]
            input_ids["decoder_attention_mask"] = decoder_input_ids["attention_mask"]
        else:
            input_ids = self.tokenizer(question + " " + answer, truncation=True, padding="max_length", return_tensors="pt")
        
        return {k: v.squeeze(0) for k, v in input_ids.items()}

    def __getitem__(self, index):
        data = self.raw_dataset[index]
        film = data["title"]
        question = f"Is {film} part of the movielens dataset? Answer only with yes or no."
        
        neg_answer, pos_answer = "No", "Yes"
        neg_ids = self.encode(question, neg_answer)
        pos_ids = self.encode(question, pos_answer)
        
        return neg_ids, pos_ids, question, neg_answer, pos_answer


def get_local_dataloader(file_path, tokenizer: PreTrainedTokenizer, batch_size=16, num_examples=1000, 
                         model_type="encoder_decoder", use_decoder=False, device="cuda", 
                         pin_memory=True, num_workers=1):
    movies = pd.read_csv(file_path, sep="::", engine="python", names=["movie_id", "title", "genre"], encoding="latin-1")
    raw_dataset = HFDataset.from_pandas(movies)
    contrast_dataset = LocalContrastDataset(raw_dataset, tokenizer, model_type=model_type, use_decoder=use_decoder, device=device)
    
    subset_dataset = torch.utils.data.Subset(contrast_dataset, list(range(min(num_examples, len(contrast_dataset)))))
    dataloader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)
    
    return dataloader

"""