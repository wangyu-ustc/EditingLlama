import random




def construct_icl_examples(idx, demos, corpus_idx):
    order = [2, 1, 2, 0, 1, 2, 2, 0, 2, 2, 1, 0, 2, 1, 2, 0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
    random.shuffle(order)
    icl_examples = []
    
    demo_ids = corpus_idx[idx]
    demo_ids = demo_ids[:len(order)]
    for demo_id, o in zip(demo_ids, order):
        
        if demo_id - 2000 >  len(demos):
            # import ipdb; ipdb.set_trace()
            continue
        line = demos[demo_id-2000]
        new_fact = line['requested_rewrite']['prompt'].format(line['requested_rewrite']['subject'])
        target_new = line['requested_rewrite']['target_new']['str']
        target_true = line['requested_rewrite']['target_true']['str']
        
        if o == 0:
            icl_examples.append(f'New Fact: {new_fact} {target_new}\nPrompt: {new_fact} {target_new}\n\n')
        elif o == 1:
            prompt = random.choice(line['paraphrase_prompts'])
            icl_examples.append(f'New Fact: {new_fact} {target_new}\nPrompt: {prompt} {target_new}\n\n')
        elif o == 2:
            prompt = random.choice(line['neighborhood_prompts'])
            icl_examples.append(f'New Fact: {new_fact} {target_new}\nPrompt: {prompt} {target_true}\n\n')
    icl_examples.reverse()
    return icl_examples