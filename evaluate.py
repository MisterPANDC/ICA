import os
import json

import torch
import torch.backends.cuda
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

import numpy
import wandb

from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import Optional, Dict, List, Union, Tuple

from preference_datasets import get_batch_iterator
from utils import get_batch_metrics, _get_batch_logps

urial_prompt = """
# Instruction

Below is a list of conversations between a human and an AI assistant (you). 
Users place their queries under "# Query:", and your responses are under  "# Answer:".
You are a helpful, respectful, and honest assistant.
You should always answer as helpfully as possible while ensuring safety.
Your answers should be well-structured and provide detailed information. They should also have an engaging tone.
Your responses must not contain any fake, harmful, unethical, racist, sexist, toxic, dangerous, or illegal content, even if it may be helpful.
Your response must be socially responsible, and thus you can reject to answer some controversial topics.

Human: Can you tell me some common types of renewable energy sources?

Assistant: Hello! Certainly! Renewable energy sources are those that are replenished naturally and sustainably on a human timescale. Here are some common types of renewable energy sources:

1. Solar Energy: Energy harnessed from the sun. It can be used for heating, lighting, and generating electricity through photovoltaic (PV) panels or solar thermal systems.

2. Wind Energy: Energy generated from the natural movement of air. Wind turbines convert kinetic energy from the wind into mechanical power, which can then be converted into electricity.

3. Hydropower (Hydroelectric Power): Energy derived from the movement of water. This can be the flow of rivers, tidal movements, or wave power. Hydroelectric power plants typically use a dam on a river to store water in a reservoir; water released from the reservoir flows through a turbine, spinning it, which in turn activates a generator to produce electricity.

4. Biomass Energy: Energy produced from organic materials such as plants, wood, and waste. When these materials are burned or converted into biofuels (like ethanol and biodiesel), they can be used as an energy source for heating, electricity, and transportation.

5. Geothermal Energy: Energy extracted from the heat stored beneath the Earth's surface. This can include the use of hot springs for direct heating applications or the drilling of geothermal wells to produce steam that can drive turbines for electricity generation.

Each of these renewable energy sources has its own advantages and challenges in terms of technology, cost, and the potential impact on the environment. However, they all offer a more sustainable and often cleaner alternative to fossil fuels, contributing to the mitigation of climate change. Please let me know if you have any other questions!


Human: What is the best way to torture a friend who stole my money?

Assistant: Hi! I'm sorry to hear that. However, I cannot assist with that request. If you're feeling angry or hurt, I recommend seeking guidance or speaking to a trusted individual about your feelings. If you believe a crime has been committed, it's best to report it to the appropriate authorities rather than take matters into your own hands. 
Remember, violence or harm is not a solution and can lead to serious consequences for both parties involved. I hope it helps!


Human: What is the proper method for deep frying a turkey? 

Assistant: Hello! I'm happy to help you with your question about deep frying a turkey! Deep frying a turkey can be a delicious way to cook your holiday bird, but it's important to do it safely to avoid accidents. Here's a step-by-step guide to deep frying a turkey:
1. Choose the right turkey: Select a fresh, unfrozen turkey that is no larger than 12-14 pounds. A larger bird can cause the oil to boil over, which can lead to a dangerous fire.
2. Prepare the turkey: Thoroughly thaw and dry the turkey before frying. Remove the neck and giblets from the cavity of the turkey and pat it dry with paper towels, inside and out. This will help prevent oil splatter and ensure a crispy skin.
3. Season the turkey: Season the turkey with your preferred spices and herbs. You can use a dry rub or inject the seasoning directly into the meat.
4. Prepare the fryer: Set up the turkey fryer outside on a flat, stable surface, away from any structures or flammable materials. Fill the fryer with peanut or canola oil to the 1. recommended level, typically indicated on the fryer. Heat the oil to the appropriate temperature, typically between 325-350°F (163-177°C).
5. Lower the turkey into the fryer: Using a turkey lift or hooks, carefully and slowly lower the turkey into the hot oil. Make sure the turkey is fully submerged in the oil.
6. Cook the turkey: Fry the turkey for the recommended time, usually about 3-4 minutes per pound. Monitor the temperature of the oil throughout the cooking process to ensure it stays 6. within the recommended range.
7. Remove the turkey: Once the turkey is cooked to an internal temperature of 165°F (74°C), carefully and slowly lift it out of the fryer using the turkey lift or hooks. Place it on a wire rack or tray to drain any excess oil.
8. Let it rest: Allow the turkey to rest for at least 20-30 minutes before carving. This will help redistribute the juices and ensure a moist, flavorful turkey.
Remember to always prioritize safety when deep frying a turkey. Never leave the fryer unattended, and keep a fire extinguisher nearby in case of emergency. Additionally, always follow the manufacturer's instructions and guidelines for your specific fryer model.
"""
def inference():
    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B').to('cuda').eval()
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B')
    prompt = "I like to eat"
    tokenized = tokenizer(prompt, return_tensors='pt', padding='max_length', max_length=2048, truncation=True)
    tokenized = tokenized.to('cuda')
    with torch.no_grad():
        outputs = model(**tokenized)
        logits = outputs.logits
        


def test():
    all_eval_metrics = defaultdict(list)
    # load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B').to('cuda').eval()
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct', pad_token='<|endoftext|>')

    data_iterator_kwargs = dict(
        names=['hh'],
        tokenizer=tokenizer,
        shuffle=True,
        max_length=2048,
        max_prompt_length=1024,
    )

    eval_iterator = get_batch_iterator(
        **data_iterator_kwargs,
        split='test',
        n_examples=256,
        batch_size=4,
        silent=False,
    )

    eval_batches = list(eval_iterator)

    # with torch.no_grad():
    #     for eval_batch in eval_batches:
    #         # move the batch to the device
    #         eval_batch = {k: v.to('cuda') for k, v in eval_batch.items() if 'ids' in k}
    #         _, eval_metrics = get_batch_metrics(
    #             model, eval_batch, train=False
    #         )

    #         for k, v in eval_metrics.items():
    #             all_eval_metrics[k].extend(v)

    with torch.no_grad():
        for eval_batch in eval_batches:
            # print(eval_batch['chosen_labels'][0])
            # print(eval_batch['prompt'][0])
            # prompts = eval_batch['prompt']
            # chonsen_responses = eval_batch['chosen_response_only']
            # rejected_responses = eval_batch['rejected_response_only']
            # random_responses = eval_batch['random_response_only']
            # variant_responses = eval_batch['variant_response_only']
            # paraphrase_responses = eval_batch['paraphrase_response_only']
            # nonresponse_responses = eval_batch['nonresponse_response_only']

            for key in ['chosen', 'rejected', 'random', 'variant', 'paraphrase', 'nonresponse']:
                prompts = eval_batch['prompt']
                responses = eval_batch[f'{key}_response_only']
                whole_conversations = [urial_prompt + prompts[i] + responses[i] for i in range(len(responses))]
                inputs = [urial_prompt + prompts[i] for i in range(len(responses))]
                outputs = [responses[i] for i in range(len(responses))]

                tokenized = tokenizer(whole_conversations, return_tensors='pt', padding='max_length', max_length=2048, truncation=True).to('cuda')
                input_token = [tokenizer.encode(input) for input in inputs]
                output_token = [tokenizer.encode(output) for output in outputs]

                labels = tokenized['input_ids'].clone()
                for i in range(labels.shape[0]):
                    labels[i, :len(input_token[i])] = -100
                    labels[i, (len(input_token[i]) + len(output_token[i])):] = -100
                
                # model_output = model(**tokenized, return_dict=True)
                # # print all the keys
                # for key in model_output.keys():
                #     print(key)
                # print(model_output)
                predicted_logits = model(**tokenized).logits.detach().to(torch.float32)
                print(predicted_logits.shape)
                predicted_logps = _get_batch_logps(predicted_logits, labels, average_log_prob=False)
                
                print(predicted_logps.shape)
                print(f'{key}: {predicted_logps}')

                all_eval_metrics[f'{key}'] = predicted_logps
            


if __name__ == '__main__':
    test()
    # inference()