#!/bin/sh
# Keep the dev container's packages in step with package-lock.json.
#
# docker-compose.yml mounts the named volume ragbot-web-node-modules over
# /app/node_modules. Docker fills a named volume from the image only when the
# volume is created, so after package-lock.json changes, a rebuilt image never
# reaches it and the dev server keeps running the old packages (on 2026-11-20 it
# ran Next.js 16.0.10 against a lockfile at 16.4.5). The image records the
# lockfile its packages came from; when the mounted lockfile differs, this
# reinstalls before starting.
set -eu

lock=/app/package-lock.json
installed=/app/node_modules/.installed-package-lock.json

if [ ! -f "$installed" ] || ! cmp -s "$lock" "$installed"; then
  echo "dev-entrypoint: package-lock.json differs from the installed packages; running npm ci"
  npm ci --no-audit --no-fund
  cp "$lock" "$installed"
fi

exec "$@"
