import collections


def dict_update(d, u):
    """Improved update for nested dictionaries.

    Arguments:
        d: The dictionary to be updated.
        u: The update dictionary.

    Returns:
        The updated dictionary.
    """
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


if __name__ == '__main__':
    default_config = {
        'primitives': 'all',
        'truncate': {},
        'validation_size': -1,
        'test_size': -1,
        'on-the-fly': False,
        'cache_in_memory': False,
        'suffix': None,
        'add_augmentation_to_test_set': False,
        'num_parallel_calls': 10,
        'generation': {
            'split_sizes': {'training': 10000, 'validation': 200, 'test': 500},
            'image_size': [960, 1280],
            'random_seed': 0,
            'params': {
                'generate_background': {
                    'min_kernel_size': 150, 'max_kernel_size': 500,
                    'min_rad_ratio': 0.02, 'max_rad_ratio': 0.031},
                'draw_stripes': {'transform_params': (0.1, 0.1)},
                'draw_multiple_polygons': {'kernel_boundaries': (50, 100)}
            },
        },
        'preprocessing': {
            'resize': [240, 320],
            'blur_size': 11,
        },
        'augmentation': {
            'photometric': {
                'enable': False,
                'primitives': 'all',
                'params': {},
                'random_order': True,
            },
            'homographic': {
                'enable': False,
                'params': {},
                'valid_border_margin': 0,
            },
        },
        'drawing_primitives': [
            'draw_lines',
            'draw_polygon',
            'draw_multiple_polygons',
            'draw_ellipses',
            'draw_star',
            'draw_checkerboard',
            'draw_stripes',
            'draw_cube',
            'gaussian_noise'
        ]
    }

    import yaml
    import json

    with open('magic-point_shapes.yaml', 'r') as f:
        config = yaml.load(f)

    config = dict_update(default_config, config['data'])

    with open('magic-point_shapes.json', 'w') as f:
        json.dump(config, f)