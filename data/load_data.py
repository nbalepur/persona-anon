import datasets

for split in ['Anthropic HHH', 'BeaverTails', 'Mnemonic', 'SHP']:  
    for name in ['persona-tailoring', 'persona-inference']:
        if split == 'SHP' and 'tailoring' in name:
            continue
        ds = datasets.load_dataset(f"nbalepur/{name}", split)
        ds.save_to_disk(f'/Users/nishantbalepur/Desktop/Repositories/persona-anon/data/{name}-{split}')