"""Discover search labels from every rendered view using local RAM++ inference."""
from collections import Counter
import gc
import json
from pathlib import Path

from PIL import Image
import torch


RAM_REPOSITORY = 'xinyu1205/recognize-anything-plus-model'
RAM_REVISION = '84d4aee3a0265c4e0df1f714f0572011d1bf2ec3'
TOKENIZER_REVISION = '86b5e0934494bd15c9632b12f734a8a67f723594'

# Scene descriptions and attributes are kept as context, not sent as object names.
# ponytail: this small filter is heuristic; a semantic taxonomy can replace it if
# non-object tags survive the subsequent grounded detector in other scene types.
CONTEXT_TAGS = {
    '3d cg rendering', 'architecture', 'bedroom', 'living room', 'interior',
    'interior design', 'indoor', 'indoors', 'room', 'furniture', 'decoration',
    'dormitory', 'office', 'home office', 'library', 'loft', 'hallway', 'stairwell', 'fill',
    'selfie', 'stare', 'catch', 'close-up', 'repair', 'draw', 'bite',
    'cloudy', 'lay', 'lead to', 'open', 'purple', 'stack', 'hardwood', 'material', 'damage',
    'wood', 'wooden', 'white', 'black', 'brown', 'green', 'blue', 'red',
    'yellow', 'colorful', 'bright', 'dark', 'sunlight', 'daylight', 'shadow',
    'cozy', 'clutter', 'cluttered', 'messy', 'clean', 'tidy', 'empty',
    'small', 'large', 'modern', 'vintage', 'old', 'standing', 'sitting',
    'meeting room', 'dining room', 'kitchen', 'restaurant kitchen', 'restaurant',
    'coffee shop', 'cafe', 'canteen', 'apartment', 'basement', 'warehouse',
}
ALIASES = {
    'bookcase': 'bookshelf', 'bookshelves': 'bookshelf', 'book shelf': 'bookshelf',
    'shelve': 'shelf',
    'potted plant': 'plant', 'houseplant': 'plant',
    'computer monitor': 'monitor', 'computer screen': 'monitor',
    'computer desk': 'desk', 'office desk': 'desk',
    'hardwood floor': 'floor', 'wood floor': 'floor', 'wall clock': 'clock',
    'office chair': 'chair', 'desk chair': 'chair', 'swivel chair': 'chair',
    'dinning table': 'table', 'dining table': 'table', 'kitchen table': 'table',
    'kitchen sink': 'sink',
}


def normalise_tag(tag):
    tag = ' '.join(tag.lower().replace('_', ' ').split())
    if not tag or tag in CONTEXT_TAGS:
        return None
    return ALIASES.get(tag, tag)


def discover_labels(frames_dir, transforms, device):
    print('[discover] Preparing tagging model', flush=True)
    from huggingface_hub import hf_hub_download, snapshot_download
    from ram import get_transform, inference_ram
    from ram.models import ram_plus

    weights = hf_hub_download(RAM_REPOSITORY, 'ram_plus_swin_large_14m.pth',
                              revision=RAM_REVISION)
    tokenizer = snapshot_download('google-bert/bert-base-uncased',
                                 revision=TOKENIZER_REVISION,
                                 allow_patterns=['tokenizer*', 'vocab.txt',
                                                 'config.json', 'special_tokens_map.json'])
    print(f'[discover] Loading RAM++ on {device}; no user object names supplied')
    model = ram_plus(pretrained=weights, image_size=384, vit='swin_l',
                     text_encoder_type=tokenizer).eval().to(device)
    transform = get_transform(image_size=384)
    counts = Counter()
    context = Counter()
    views = []
    vocabulary_size = int(model.num_class)
    try:
        for index, frame in enumerate(transforms['frames']):
            path = Path(frames_dir).parent / frame['file_path']
            with Image.open(path) as image:
                tensor = transform(image).unsqueeze(0).to(device)
            tags = [tag.strip() for tag in inference_ram(tensor, model)[0].split('|')
                    if tag.strip()]
            names = sorted({name for tag in tags if (name := normalise_tag(tag))})
            counts.update(names)
            context.update(tag for tag in tags if normalise_tag(tag) is None)
            views.append({'frame_idx': index, 'image': frame['file_path'],
                          'tags': tags, 'object_labels': names})
            if (index + 1) % 10 == 0 or index + 1 == len(transforms['frames']):
                print(f'[discover] {index + 1}/{len(transforms["frames"])} views; '
                      f'{len(counts)} labels discovered')
    finally:
        del model
        gc.collect()
        if device == 'mps':
            torch.mps.empty_cache()
        elif device.startswith('cuda'):
            torch.cuda.empty_cache()

    labels = sorted(counts, key=lambda name: (-counts[name], name))
    report = {
        'mode': 'automatic', 'model': 'RAM++', 'model_repository': RAM_REPOSITORY,
        'model_revision': RAM_REVISION, 'device': device,
        'vocabulary_size': vocabulary_size, 'user_supplied_labels': False,
        'labels': labels, 'supporting_view_counts': dict(counts),
        'context_tags': dict(context), 'views': views,
        'limitations': ['Uses the pretrained image tagger vocabulary; unseen or obscured objects can be missed.'],
    }
    (Path(frames_dir).parent / 'discovery.json').write_text(json.dumps(report, indent=2) + '\n')
    return labels, report
