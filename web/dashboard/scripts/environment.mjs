// SPDX-License-Identifier: AGPL-3.0-only
import {existsSync, realpathSync} from 'node:fs';
import {fileURLToPath} from 'node:url';

export function assertManagedContainer() {
  const directory = fileURLToPath(new URL('..', import.meta.url));
  if (process.platform !== 'linux' || !existsSync('/.dockerenv')
      || realpathSync(directory) !== '/mityane/web/dashboard') {
    throw new Error('Use container.ps1 and the managed mityane-geometrics container at /mityane.');
  }
}
