import argparse
import datasets
import json

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_name",
        type=str,
        help="String to identify this run",
        default="default",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Name of the model",
        default="gpt-4o",
    )
    parser.add_argument(
        "--inference_split",
        type=str,
        help="Split of the dataset",
        default="BeaverTails",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Local path or huggingface path of the original dataset",
        default="nbalepur/persona-inference",
    )
    parser.add_argument(
        "--new_dataset_name",
        type=str,
        help="Name of the new dataset",
        default="",
    )
    parser.add_argument(
        "--hf_write_token",
        type=str,
        help="Huggingface write token",
        default="",
    )
    parser.add_argument(
        "--push_to_hub",
        action='store_true',
        help="Should we push the model to the hub?",
        default=True,
    )

    # load existing data
    args = parser.parse_args()
    ds = datasets.load_dataset(args.dataset_name, args.inference_split)['train']

    f = f'results/{args.model}/{args.inference_split}/{args.run_name}/persona_inference.jsonl'
    with open(f, 'r') as json_file:
        json_list = list(json_file)

    assert len(json_list) == (2 * ds.num_rows), "Your number of personas doesn't match the size of the dataset (did you run persona inference on this full dataset?)"

    # collect chosen + rejected personas and add them to the dataset
    chosen = [json.loads(t)['raw_text'].split('\n')[0].strip() for t in json_list[:ds.num_rows]]
    rejected = [json.loads(t)['raw_text'].split('\n')[0].strip() for t in json_list[ds.num_rows:]]

    # upload new dataset
    new_ds = ds.add_column('chosen_persona_gold', chosen).add_column('rejected_persona_gold', rejected)
    if args.push_to_hub:
        new_ds.push_to_hub(args.new_dataset_name, token=args.hf_write_token)
    else:
        new_ds.save_to_disk(args.new_dataset_name, token=args.hf_write_token)

if __name__ == '__main__':
    main()