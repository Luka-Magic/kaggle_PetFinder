def make_columns(columns):
    return_columns = []
    if 'original' in columns or 'all' in columns:
        return_columns.extend(['Subject Focus', 'Eyes', 'Face', 'Near', 'Action',
                               'Accessory', 'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur'])
    if 'basic' in columns or 'all' in columns:
        return_columns.extend(
            ['height', 'width', 'size', 'sqrtsize', 'aspect'])
    if 'hash' in columns or 'all' in columns:
        return_columns.extend([f'hash_{i}' for i in range(256)])
    return return_columns


def len_columns(columns):
    len_column = 0
    if 'original' in columns or 'all' in columns:
        len_column += 12
    if 'basic' in columns or 'all' in columns:
        len_column += 5
    if 'hash' in columns or 'all' in columns:
        len_column += 256
    return len_column
