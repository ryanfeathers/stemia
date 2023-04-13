from pathlib import Path, PureWindowsPath
import mdocfile
from xml.etree import ElementTree


def parse_data(progress, warp_dir, mdoc_dir, output_dir, roi_dir, just=(), exclude=(), train=False):
    imod_dir = warp_dir / 'imod'
    if not imod_dir.exists():
        raise FileNotFoundError('warp directory does not have an `imod` subdirectory')

    if just:
        mdocs = [p for ts_name in just if (p := (Path(mdoc_dir) / (ts_name + '.mdoc'))).exists()]
    else:
        mdocs = sorted(list(Path(mdoc_dir).glob('*.mdoc')))

    if not mdocs:
        raise FileNotFoundError('could not find any mdoc files')

    odd_dir = warp_dir / 'average' / 'odd'
    even_dir = warp_dir / 'average' / 'even'

    tilt_series = []
    tilt_series_excluded = []
    tilt_series_unprocessed = []

    for mdoc in progress.track(mdocs, description='Reading mdocs...'):
        ts_name = mdoc.stem
        stack = imod_dir / ts_name / (ts_name + '.st')

        if ts_name in exclude:
            tilt_series_excluded.append(ts_name)
            continue
        # skip if not preprocessed in warp
        if not stack.exists():
            tilt_series_unprocessed.append(ts_name)
            continue

        df = mdocfile.read(mdoc)

        # extract even/odd paths
        tilts = [warp_dir / PureWindowsPath(tilt).name for tilt in df.SubFramePath]
        odd = []
        even = []
        valid_xml = None
        for tilt in tilts:
            xml = ElementTree.parse(tilt.with_suffix('.xml')).getroot()
            if xml.attrib['UnselectManual'] == 'True':
                continue

            valid_xml = xml
            odd.append(odd_dir / (tilt.stem + '.mrc'))
            even.append(even_dir / (tilt.stem + '.mrc'))

        if valid_xml is None:
            continue

        if train:
            for img in odd + even:
                if not img.exists():
                    raise FileNotFoundError(img)

        # extract metadata from warp xmls (we assume the last xml has the same data as the others)
        for param in valid_xml.find('OptionsCTF'):
            if param.get('Name') == 'BinTimes':
                bin = float(param.get('Value'))
            elif param.get('Name') == 'Voltage':
                kv = int(param.get('Value'))
            elif param.get('Name') == 'Cs':
                cs = float(param.get('Value'))
        for param in xml.find('CTF'):
            if param.get('Name') == 'Defocus':
                defocus = float(param.get('Value'))

        if roi_dir is not None:
            roi_files = list(roi_dir.glob(f'{ts_name}*'))
            if len(roi_files) == 1:
                roi_file = roi_files[0]
            else:
                roi_file = None
        else:
            roi_file = None

        # due to a quirk of aretomo, with_suffix is wrong in some cases
        # because *all* extensions are removed
        chopped_ts_name = ts_name.split('.')[0]

        tilt_series.append({
            'name': ts_name,
            'stack': stack,
            'rawtlt': stack.with_suffix('.rawtlt'),
            'fix': output_dir / (ts_name + '_fix.st'),
            'aln': output_dir / (chopped_ts_name + '.aln'),
            'xf': output_dir / (chopped_ts_name + '_Imod') / (chopped_ts_name + '.xf'),
            'roi': roi_file,
            'odd': odd,
            'even': even,
            'stack_odd': output_dir / (ts_name + '_odd.st'),
            'stack_even': output_dir / (ts_name + '_even.st'),
            'recon_odd': output_dir / 'odd' / (ts_name + '.mrc'),
            'recon_even': output_dir / 'even' / (ts_name + '.mrc'),
            'recon': output_dir / (ts_name + '.mrc'),
            'aretomo_kwargs': {
                'dose': df.ExposureDose[0],
                'px_size': df.PixelSpacing[0] * 2**bin,
                'cs': cs,
                'kv': kv,
                'defocus': defocus,
            }
        })

    return tilt_series, tilt_series_excluded, tilt_series_unprocessed
