run_name = 'default' # name of the run
evaluator_models = ['gpt-4o-mini'] # models that judged
models = ['gpt-4o-mini'] # models that were judged
datasets = ['BeaverTails'] # name of the datasets

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
import matplotlib.patches as mpatches
from typing import Any, List
import json
import re

acc_data = {'evalutor_model': [], 'Model': [], 'correct': [], 'Response Type': [], 'dataset': []}

def clf(output: Any) -> int:
    """Parse the model's classification response"""
    if type(output) != type(''):
        return -1
    if "1" in output and "2" not in output:
        return 1
    if "2" in output and "1" not in output:
        return 2
    return -1

def clf_pref(judgments: List[int]) -> str:
    """Return which persona the model believed was better"""
    if judgments[0] + judgments[1] != 3:
        return 'Tie'
    return 'Chosen Persona' if judgments[0] == 1 else 'Rejected Persona'

custom_palette = {"Tie": "blue", "Chosen Persona": "green", "Rejected Persona": "red"}

pref_persona_ds = {'model': [], 'dataset': [], 'preference': []}

for m in evaluator_models:
    for ds in datasets:

        f = f'results/{m}/{ds}/{run_name}/persona_prefs.jsonl'
        with open(f, 'r') as json_file:
            json_no_persona = list(json_file)

        pref_persona_dict = dict()
        for l in json_no_persona:
            l = json.loads(l)
            label = clf(output=l['raw_text'])

            persona_regex = r"Persona 1: (.+?)\n---\nPersona 2: (.+?)\n---\nBetter Persona:"
            match = re.search(persona_regex, l['prompt'])
            if match:
                persona1 = match.group(1).strip()
                persona2 = match.group(2).strip()

                personas_key = tuple(list(sorted([persona1, persona2])))

                arr = pref_persona_dict.get(personas_key, [])
                arr.append(label)
                pref_persona_dict[personas_key] = arr

        pref_persona_dict = {k: clf_pref(judgments=v) for k, v in pref_persona_dict.items() if len(v) == 2 and -1 not in v}

        for v in pref_persona_dict.values():
            pref_persona_ds['model'].append(MODEL_MAP.get(m, m))
            pref_persona_ds['dataset'].append(DATASET_MAP.get(ds, ds))
            pref_persona_ds['preference'].append(v)

# First dataframe (no persona)
df1 = pd.DataFrame(pref_persona_ds)

for dataset in set(df1['dataset']):
    curr_df = df1[df1['dataset'] == dataset]
    x = curr_df['preference'].value_counts() / len(curr_df)

df1['model'] = pd.Categorical(df1['model'])
df1['dataset'] = pd.Categorical(df1['dataset'])

# Grouping and calculating proportions for df1
grouped_df1 = df1.groupby(['dataset', 'model', 'preference']).size().reset_index(name='count')
grouped_df1['proportion'] = grouped_df1.groupby(['dataset', 'model'])['count'].transform(lambda x: x / x.sum())

# Get unique datasets and models
datasets = grouped_df1['dataset'].unique()

# Define custom colors for each preference category
custom_palette = {'Chosen Persona': '#4565ae', 'Rejected Persona': '#cd2428', 'Tie': '#c5c6c7'}
hatch_map = {'Chosen Persona': '///', 'Rejected Persona': '\\\\\\', 'Tie': ''}

# Create a 2x2 figure layout
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(14, 2.5))
axs = axes.flatten()  # Flatten axes array for easy indexing

# Loop through each dataset and create a subplot in the 2x2 layout
for i, dataset in enumerate(datasets):
    if i < 4:  # Limit to the first four datasets if there are more
        ax = axs[i]
        subset = grouped_df1[grouped_df1['dataset'] == dataset]
        
        # Initialize bottom of the bar to stack values
        bottom = np.zeros(len(models))
        
        # Plot each preference category as a stacked segment
        for preference in ['Chosen Persona', 'Tie', 'Rejected Persona']:
            # Filter data for the current preference
            preference_data = subset[subset['preference'] == preference]
            
            # Ensure data is aligned with models for stacking
            proportions = [preference_data[preference_data['model'] == model]['proportion'].values[0]
                           if model in preference_data['model'].values else 0
                           for model in models]
            
            # Create the bar segment for the current preference
            ax.bar(models, proportions, bottom=bottom, color=custom_palette[preference], label=preference, hatch=hatch_map[preference])
            bottom += np.array(proportions)  # Update bottom for next stack

        # Title and axis settings
        ax.set_title(dataset)
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)

# Remove individual legends
for ax in axs:
    ax.legend().remove()

# Create proxy artists for the single legend
legend_patches = [mpatches.Patch(color=color, label=label) for label, color in custom_palette.items()]

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3)

# Adjust layout to fit the legend at the bottom
plt.tight_layout()
fig.subplots_adjust(bottom=0.40)
plt.savefig('plots/images/persona_prefs.pdf', format='pdf')