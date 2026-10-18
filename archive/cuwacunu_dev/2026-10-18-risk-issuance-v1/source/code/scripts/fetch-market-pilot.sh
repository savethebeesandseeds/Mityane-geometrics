#!/usr/bin/env bash
# Public data bootstrap only; dependencies/container lifecycle stay elsewhere.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
[[ -f /.dockerenv && "$(pwd -P)" == /cuwacunu ]] || {
  echo 'Run in the managed cuwacunu container.' >&2; exit 1;
}
[[ $# == 0 ]] || { echo 'Usage: bash code/scripts/fetch-market-pilot.sh' >&2; exit 1; }
destination=data/mkt-hist-1
if [[ -e "$destination" ]]; then
  [[ -f "$destination/manifest.json" ]] || {
    echo 'Existing data directory has no complete manifest; preserved.' >&2; exit 1;
  }
  python3 - "$destination" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); manifest=json.loads((root/'manifest.json').read_text())
assert manifest['dataset_id']=='MKT-HIST-1'
for entry in manifest['files']:
    for kind in ('zip','csv'):
        path=root/entry[kind+'_path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==entry[kind+'_sha256'],str(path)
print('Reused verified MKT-HIST-1 data; no download.')
PY
  exit 0
fi
mkdir -p data
staging=$(mktemp -d data/mkt-hist-1-fetch-XXXXXX)
mkdir "$staging/archives" "$staging/csv"
printf 'Preserved download staging: %s\n' "$staging"
for symbol in BTCUSDT ETHUSDT SOLUSDT; do
  name="$symbol-1m-2024-01.zip"
  url="https://data.binance.vision/data/spot/monthly/klines/$symbol/1m/$name"
  curl --fail --location --proto '=https' --tlsv1.2 --retry 2 --max-time 120 \
    --output "$staging/archives/$name" "$url"
  curl --fail --location --proto '=https' --tlsv1.2 --retry 2 --max-time 120 \
    --output "$staging/archives/$name.CHECKSUM" "$url.CHECKSUM"
  (cd "$staging/archives" && sha256sum -c "$name.CHECKSUM")
done
# Standard-library extraction avoids installing another archive tool. Only the
# exact numeric CSV member is accepted; no paths/code from an archive execute.
python3 - "$staging" <<'PY'
import csv,hashlib,json,pathlib,sys,zipfile
root=pathlib.Path(sys.argv[1]); entries=[]
for symbol in ('BTCUSDT','ETHUSDT','SOLUSDT'):
    stem=f'{symbol}-1m-2024-01'; archive=root/'archives'/f'{stem}.zip'
    with zipfile.ZipFile(archive) as z:
        assert z.namelist()==[f'{stem}.csv'],z.namelist()
        info=z.getinfo(f'{stem}.csv'); assert info.file_size<50_000_000
        payload=z.read(info)
    target=root/'csv'/f'{stem}.csv'
    with target.open('xb') as out: out.write(payload)
    count=0; first=None; last=None; previous=None
    with target.open(newline='') as source:
        for row in csv.reader(source):
            assert len(row)==12,'Twelve-column spot kline schema required'
            key=int(row[6]); opening=int(row[0])
            assert key==opening+59999
            assert previous is None or key==previous+60000,'Unexpected grid gap'
            first=key if first is None else first; previous=last=key; count+=1
    assert (count,first,last)==(44640,1704067259999,1706745599999),(count,first,last)
    entries.append(dict(symbol=symbol,interval='1m',month='2024-01',
        source_url=f'https://data.binance.vision/data/spot/monthly/klines/{symbol}/1m/{stem}.zip',
        checksum_url=f'https://data.binance.vision/data/spot/monthly/klines/{symbol}/1m/{stem}.zip.CHECKSUM',
        zip_path=f'archives/{stem}.zip',csv_path=f'csv/{stem}.csv',
        zip_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        csv_sha256=hashlib.sha256(payload).hexdigest(),rows=count,
        first_close_ms=first,last_close_ms=last))
with (root/'manifest.json').open('x') as out:
    json.dump(dict(dataset_id='MKT-HIST-1',kind='public-binance-spot-history-v1',
        timestamp_units='milliseconds',files=entries),out,indent=2); out.write('\n')
print('Verified three complete January 2024 one-minute archives.')
PY
[[ ! -e "$destination" ]] || { echo 'Destination appeared; staging preserved.' >&2; exit 1; }
mv -T --no-clobber "$staging" "$destination"
[[ -f "$destination/manifest.json" ]] || { echo 'Publication failed; data preserved.' >&2; exit 1; }
printf 'Public data ready: %s/manifest.json\n' "$destination"
