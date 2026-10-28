"""Local sentence embeddings for positive subject paraphrases; no generated IDs."""
import re

import numpy as np

from semantic_binding import catalog, compatible_category

MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
MIN_SIMILARITY = .50  # Cosine retrieval heuristic, not a calibrated probability.
COLORS = {'red','orange','yellow','green','blue','purple','pink','brown','black','white','gray','grey'}
FILLER = {'show','find','focus','on','off','my','me','please','the','a','an','at','look','toward','towards'}


class SemanticSearch:
    def __init__(self):
        import torch
        from transformers import AutoTokenizer, BertModel
        self.torch = torch
        torch.set_num_threads(min(4, torch.get_num_threads()))
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        self.model = BertModel.from_pretrained(MODEL, revision=REVISION).eval()

    def encode(self, texts):
        outputs = []
        for start in range(0, len(texts), 32):
            inputs = self.tokenizer(texts[start:start+32], padding=True, truncation=True, max_length=128, return_tensors='pt')
            with self.torch.inference_mode():
                tokens = self.model(**inputs)[0]
                mask = inputs['attention_mask'].unsqueeze(-1).float()
                pooled = (tokens*mask).sum(1)/mask.sum(1).clamp(min=1e-9)
                outputs.append(self.torch.nn.functional.normalize(pooled, p=2, dim=1).numpy())
        return np.concatenate(outputs)

    def search(self, index, prompt):
        if re.search(r"\b(not|except|without|avoid|exclude)\b|don't", prompt.lower()):
            return dict(target_groups=[], target_categories=[], message='Try a positive subject request; exclusion queries are not supported yet.')
        if re.search(r'\b(above|below|behind|between|left|right)\b', prompt.lower()):
            return dict(target_groups=[],target_categories=[],message='Directional and multi-step spatial queries are not supported yet; try a subject or a nearby-object request.')
        units = []
        texts = []
        for group in index['groups']:
            for text in [group['name'], group['name']+'. '+group['description']]:
                texts.append(text)
                units.append(('group', group['name']))
        for category, examples in catalog(index['objects']).items():
            for text in [category, category+'. '+'. '.join(examples)]:
                texts.append(text)
                units.append(('category', category))
        if not texts:
            return dict(target_groups=[], target_categories=[])
        words = re.findall(r'\w+', prompt.lower())
        query = ' '.join(word for word in words if word not in FILLER) or prompt
        vectors = self.encode(texts+[query])
        scores = vectors[:-1] @ vectors[-1]
        best = int(np.argmax(scores))
        explicit = {category for category in catalog(index['objects']) if re.search(r'\b'+re.escape(category)+r's?\b', query)}
        if explicit:
            candidates = [i for i,(kind,name) in enumerate(units) if kind=='category' and name in explicit]
            best = max(candidates, key=lambda i:scores[i])
        else:
            group_candidates = [i for i,(kind,_) in enumerate(units) if kind=='group' and scores[i]>=MIN_SIMILARITY]
            if group_candidates:
                group_best=max(group_candidates,key=lambda i:scores[i])
                if scores[group_best]>=scores[best]-.18:
                    best=group_best
        kind, target = units[best]
        score = float(scores[best])
        routed_from = None
        # A station/area request concerns a functional group containing the
        # matched subject type (e.g. a workstation belongs to a desk setup).
        if kind=='category' and not explicit and score>=MIN_SIMILARITY and re.search(r'(station|space|area|corner|setup)\b', query):
            by_id={o['id']:o for o in index['objects']}
            head=target.split()[-1]
            compatible_names={g['name'] for g in index['groups'] if any(
                compatible_category(by_id[key]['category'], target) or compatible_category(by_id[key]['category'], head)
                for key in g['object_ids'])}
            options=[i for i,(unit_kind,name) in enumerate(units) if unit_kind=='group' and name in compatible_names]
            if options:
                routed_from=target
                group_best=max(options,key=lambda i:scores[i])
                kind,target=units[group_best]

        result = dict(target_groups=[], target_categories=[], near_category='', attributes='', tour=False,
                      match=dict(method='sentence_embedding', model=MODEL, revision=REVISION,
                                 similarity=round(score, 4), threshold=MIN_SIMILARITY))
        if routed_from:
            result['match']['routed_from_category']=routed_from
        if score < MIN_SIMILARITY:
            return result
        colors = set(words) & COLORS
        result['attributes'] = ' '.join(sorted(colors))
        if kind == 'group':
            if colors:
                group = next(g for g in index['groups'] if g['name'] == target)
                names = ' '.join(o['display_name'].lower() for o in index['objects'] if o['id'] in group['object_ids'])
                if not all(color in names for color in colors):
                    return result
            result['target_groups'] = [target]
        else:
            result['target_categories'] = [target]
        if score < .6:
            result['message'] = 'Approximate semantic match; check the selected subjects.'
        return result


def semantic_intent(index, prompt):
    return SemanticSearch().search(index, prompt)
