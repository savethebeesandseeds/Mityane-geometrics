#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Record prospective work without changing any measured result or leader."""
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path('/cuwacunu')
CARD = 'f620298e9ecf9f73a32ae61737b67295f8dcde709416aa3d6ed16feee613e45c'
EXPECTED = '7ab51a5beec2eb516bf85d251931eb1b56e3f8622c7199fca2cad8202989aab4'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert Path('/.dockerenv').is_file() and Path.cwd().resolve() == ROOT
    card = ROOT / 'doc/MARKET_LOCATION_SCALE_DECODER_CARD.md'
    assert sha(card) == CARD
    path = ROOT / 'doc/encoder_search_ledger.json'
    md = ROOT / 'doc/ENCODER_SEARCH_LEDGER.md'
    assert sha(path) == EXPECTED
    assert sha(md) == '92da224cd5a30a2ac0615388a27fafd4718b8fa45275f57d4b016ec33fbab1fb'
    value = json.loads(path.read_text())
    assert len(value['candidate_entries']) == 10
    before = copy.deepcopy(value)
    evidence = ROOT / 'doc/results/market_location_scale_decoder_v1'
    evidence.mkdir(exist_ok=True)
    for src, name in ((path, 'search-ledger-before.json'), (md, 'search-ledger-before.md')):
        with (evidence / name).open('xb') as stream:
            stream.write(src.read_bytes())
    value['live']['active_work'] = ('RPB-v30.market-01 prospectively declared: Gaussian waveform location/scale '
                                  'decoder with learned CF variance and intercept; admission and quality pending')
    value['live']['active_work_is_new_encoder_candidate'] = True
    value['live']['active_candidate_card'] = dict(path=card.relative_to(ROOT).as_posix(), sha256=CARD,
                                                status='declared_before_admission_and_quality')
    assert value['candidate_entries'] == before['candidate_entries']
    assert value['leader_views'] == before['leader_views']
    assert value['leader_history'] == before['leader_history']
    assert value['live']['primary_architecture'] == 'RPB-v22'
    assert value['live']['leading_balanced_research_reference'] == 'RPB-v24.market-01.alt-01'
    text = json.dumps(value, indent=2, allow_nan=False) + '\n'
    path.write_text(text)
    note = ('\nRPB-v30.market-01 is separately declared in '
            '[its prospective card](MARKET_LOCATION_SCALE_DECODER_CARD.md), SHA256 ' + CARD + '. '
            'Admission and quality are pending; this declaration adds no measured result, '
            'changes no leader, and preserves all10 entries/50 checkpoint pins.\n')
    original = md.read_text()
    needle = '## Protection and next direction\n'
    assert original.count(needle) == 1
    md.write_text(original.replace(needle, needle + note))
    receipt = dict(kind='market_location_scale_decoder_declaration_v1', status='passed',
                   card_sha256=CARD, prior_ledger_sha256=EXPECTED,
                   ledger_sha256=sha(path), markdown_sha256=sha(md), measured_entries_unchanged=10,
                   leader_views_unchanged=True, primary_unchanged=True, balanced_unchanged=True,
                   new_encoder_calls=0, new_encoder_updates=0, new_head_fits=0, new_calibration_fits=0)
    with (evidence / 'declaration.json').open('x') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
