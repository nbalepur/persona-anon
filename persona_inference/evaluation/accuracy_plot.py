run_name = 'default'  # which run name?
evaluator_models = ['gpt-4o-mini'] # what are the evaluator models?
datasets = ['BeaverTails'] # which datasets?

# use this to rename models/datasets (optional)
MODEL_MAP = {
    'gpt-4o-mini': 'GPT-4o Mini',
}
DATASET_MAP = {
    'BeaverTails': 'Beaver Tails' 
}

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

acc_data = {'evalutor_model': [], 'Model': [], 'correct': [], 'Response Type': [], 'dataset': [], 'prompt': [], 'prediction': []}

def ae(pred: str, true: str) -> int:
    """Answer equivalence -> return a score for if the two judgments are the same"""
    if '1' in pred and '1' in true and '2' not in pred and '2' not in true:
        return 1
    if '2' in pred and '2' in true and '1' not in pred and '1' not in true:
        return 1    
    return 0


COLOR_MAP = {'Chosen Response': '#4565ae', 'Rejected Response': '#cd2428'}
HATCH_MAP = {'Chosen Response': '///', 'Rejected Response': ''}

import json
for m in evaluator_models:
    for ds in datasets:
        
        f = f'results/{m}/{ds}/{run_name}/persona_accuracy_key.jsonl'
        with open(f, 'r') as json_file:
            json_label_list = list(json_file)

        f = f'results/{m}/{ds}/{run_name}/persona_accuracy.jsonl'
        with open(f, 'r') as json_file:
            json_pred_list = list(json_file)


        for idx in range(len(json_pred_list)):

            p, l = json_pred_list[idx], json_label_list[idx]
            p, l = json.loads(p), json.loads(l)
    
            if p['raw_text'] == None:
                continue
            
            acc_data['dataset'].append(DATASET_MAP.get(ds, ds))
            acc_data['Response Type'].append('Chosen Response' if l['is_chosen'] else 'Rejected Response')
            acc_data['evalutor_model'].append(MODEL_MAP.get(m, m))
            acc_data['Model'].append(MODEL_MAP.get(l['model_name'], l['model_name']))
            acc_data['correct'].append(ae(pred=p['raw_text'].strip(), true=str(l['label'])))
            acc_data['prompt'].append(p['prompt'])
            acc_data['prediction'].append(p['raw_text'])

df = pd.DataFrame(acc_data)


acc_df = df.groupby(['Model', 'Response Type', 'dataset'])['correct'].agg(['mean', 'sem']).reset_index()
acc_df.rename(columns={'mean': 'accuracy', 'sem': 'error'}, inplace=True)

# Getting the unique datasets
unique_datasets = acc_df['dataset'].unique()

# Setting up the subplots with enough space between them
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 3.5))
axes = axes.flat

# Loop through each dataset to create a subplot
for i, dataset in enumerate(unique_datasets):
    subset = acc_df[acc_df['dataset'] == dataset]
    
    # Pivoting the data to create a grouped bar chart with error bars
    pivot_data = subset.pivot(index='Model', columns='Response Type', values='accuracy')
    pivot_error = subset.pivot(index='Model', columns='Response Type', values='error')
    
    # Bar positions and width
    bar_width = 0.35
    x = np.arange(len(pivot_data.index))

    # Plotting the grouped bar chart with error bars
    for j, response_type in enumerate(pivot_data.columns):
        axes[i].bar(
            x + j * bar_width,
            pivot_data[response_type],
            yerr=pivot_error[response_type],
            width=bar_width,
            label=response_type,
            color=COLOR_MAP[response_type],
            hatch=HATCH_MAP[response_type],
            capsize=4
        )
    
    # Set title and labels
    axes[i].set_title(dataset)
    if i % 2 == 0:
        axes[i].set_ylabel('Accuracy')
    else:
        axes[i].set_ylabel('')
    axes[i].set_xlabel('')
    
    # Rotate x-axis labels and set x-ticks
    axes[i].set_xticks(x + bar_width / 2)
    axes[i].set_xticklabels(pivot_data.index)
    axes[i].tick_params(axis='x', rotation=0)
    
    # Remove individual legends
    axes[i].set_ylim(bottom=0.0, top=1.0) 
    axes[i].axhline(y=0.5, color='orange', linestyle='--', label='Random Baseline') 

# Unified legend at the bottom
fig.legend(['Random Baseline', 'Chosen Response', 'Rejected Response'], loc='lower center', ncol=3)

# Adjust layout and save
plt.tight_layout()
plt.subplots_adjust(bottom=0.20)
plt.savefig('plots/images/accuracy_plot.pdf', format='pdf')